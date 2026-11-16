// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;
use std::task::{Context, Poll};

use crate::morsel::{MorselStream, Morselizer};
use arrow::record_batch::RecordBatch;
use datafusion_common::{DataFusionError, Result};
use datafusion_physical_plan::metrics::ScopedTimerGuard;
use futures::StreamExt as _;

use super::work_pool::{PoolConfig, PoolStep, SharedPool, UnitId, WorkPool};
use super::work_source::WorkSource;
use super::{FileStreamMetrics, OnError, StartableTime};

pub(super) use super::work_pool::ReadOrder;

/// State [`FileStreamState::Scan`].
///
/// There is one `ScanState` per `FileStream`, and thus per output partition.
///
/// It drives the lifecycle of scanning files: unopened files (from the
/// [`WorkSource`]), CPU-ready planners, planner I/O, ready morsels and the
/// active reader. Everything but the active reader lives in a [`WorkPool`],
/// which is private to this stream, or — for scans that share work between
/// partitions — shared with all sibling streams.
///
/// # I/O budget
///
/// Without a budget, every stream has one unit of work (an I/O in flight, or
/// a morsel ready or being read): each partition alternates between fetching
/// and decoding.
///
/// With a budget, I/O is issued ahead of decoding (the next row groups of the
/// current file, further files) as long as
///
/// ```text
/// bytes of in-flight I/O + bytes buffered in ready morsels and readers <= budget
/// ```
///
/// When the budget is used up, one I/O at a time may still start in the whole
/// query, so that it makes progress. I/O futures are lazy and only started
/// once admitted. Started I/O is spawned onto the Tokio runtime (when
/// available) so it makes progress while streams decode. The budget is shared
/// by all scans of the query (see [`IoBudget`]).
///
/// [`IoBudget`]: datafusion_execution::IoBudget
///
/// # Sharing work
///
/// With a shared pool, prefetching decides *when* data is fetched, not *who*
/// reads it: any stream can plan any planner, collect any I/O and read any
/// fetched morsel, and a stream only finishes when no work is left anywhere
/// (see [`WorkPool`]). Without this, work one stream claimed to prefetch it
/// could not be balanced, and the query would wait for the most loaded
/// stream (observed on local SSD), or a stream could finish while work it
/// could have taken was still in flight.
///
/// # Ordering
///
/// See [`ReadOrder`].
///
/// # State Transitions
///
/// ```text
/// work_source
///    |
///    v
/// morselizer.plan_file(file)
///    |
///    v
/// ready_planners ---> plan() ---> ready_morsels ---> into_stream() ---> reader ---> RecordBatches
///       ^               |
///       |               v
///       |          pending I/O (budgeted)
///       |               |
///       |               v
///       +-------- I/O completes
/// ```
///
/// [`FileStreamState::Scan`]: super::FileStreamState::Scan
pub(super) struct ScanState {
    /// Unopened files that still need to be planned.
    work_source: WorkSource,
    /// Remaining row limit, if any.
    remain: Option<usize>,
    /// The morselizer used to plan files.
    morselizer: Box<dyn Morselizer>,
    /// Behavior if opening or scanning a file fails.
    on_error: OnError,
    /// Outstanding work (private, or shared with sibling streams).
    pool: SharedPool,
    /// This stream's id when waiting for work of `pool`.
    waiter: usize,
    /// The morsel this stream is reading, if any.
    reader: Option<ActiveReader>,
    /// Whether this stream spawned I/O since it last yielded (see
    /// `poll_scan`).
    yield_before_read: bool,
    /// Set once the work source has no more files.
    source_done: bool,
    /// Metrics for the active scan queues.
    metrics: FileStreamMetrics,
}

struct ActiveReader {
    unit: UnitId,
    bytes: usize,
    stream: MorselStream,
}

fn start_timer(time: &mut StartableTime) {
    if time.start.is_none() {
        time.start();
    }
}

impl ScanState {
    /// Create a new scan state.
    ///
    /// The pool is shared with sibling streams when they share a work source,
    /// there is no limit and order is not [`ReadOrder::Required`];
    /// otherwise it is private.
    pub(super) fn new(
        work_source: WorkSource,
        remain: Option<usize>,
        morselizer: Box<dyn Morselizer>,
        on_error: OnError,
        metrics: FileStreamMetrics,
        mut config: PoolConfig,
    ) -> Self {
        if remain.is_some() {
            // avoid reading ahead data that will not be needed
            config.io_budget = None;
        }
        // Siblings share their outstanding work unless the order matters (or
        // a limit applies, whose file accounting is per stream).
        let shared = work_source
            .shared()
            .filter(|_| remain.is_none() && config.read_order != ReadOrder::Required);
        let streams = if shared.is_some() {
            config.streams.max(1)
        } else {
            1
        };
        // A pool driven by a single stream reads in planned order: reading
        // whatever is ready first only pays off when several streams share the
        // work, and a single partition then returns rows in file order (as
        // without prefetch) even though its I/O completes in any order.
        if streams == 1 && config.read_order == ReadOrder::Any {
            config.read_order = ReadOrder::Preferred;
        }
        let config = PoolConfig { streams, ..config };
        let pool = match shared {
            Some(shared) => shared.pool(config),
            None => WorkPool::new_shared(config),
        };
        let waiter = pool.lock().new_waiter();
        Self {
            work_source,
            remain,
            morselizer,
            on_error,
            pool,
            waiter,
            reader: None,
            yield_before_read: false,
            source_done: false,
            metrics,
        }
    }

    /// Updates how scan errors are handled while the stream is still active.
    pub(super) fn set_on_error(&mut self, on_error: OnError) {
        self.on_error = on_error;
    }

    fn skip_errors(&self) -> bool {
        matches!(self.on_error, OnError::Skip)
    }

    fn step_result(step: PoolStep) -> Option<ScanAndReturn> {
        match step {
            PoolStep::Progress => Some(ScanAndReturn::Continue),
            PoolStep::Idle => None,
            PoolStep::Error(err) => Some(ScanAndReturn::Error(err)),
        }
    }

    /// Drives one iteration of the active scan state.
    ///
    /// Work is attempted in this order:
    /// 1. collect completed planner I/O
    /// 2. start planner I/O that fits within the I/O budget
    /// 3. run CPU planning on a ready planner
    /// 4. poll the active reader
    /// 5. turn a ready morsel into the active reader
    /// 6. open the next unopened file
    ///
    /// The return [`ScanAndReturn`] tells `poll_inner` how to update the
    /// outer `FileStreamState`.
    pub(super) fn poll_scan(&mut self, cx: &mut Context<'_>) -> ScanAndReturn {
        let time_processing = self.metrics.time_processing.clone();
        let _timer: ScopedTimerGuard<'_> = time_processing.timer();
        let skip_errors = self.skip_errors();
        let reading = self.reader.is_some();
        let pool = Arc::clone(&self.pool);

        // 1.-3.: I/O and planning
        {
            let mut guard = pool.lock();
            guard.pass_on(self.waiter);
            if let Some(ret) =
                Self::step_result(guard.collect_io(cx, &mut self.metrics, skip_errors))
            {
                return ret;
            }
            if guard.start_io(&pool) {
                self.yield_before_read = true;
                return ScanAndReturn::Continue;
            }
            if let Some((unit, planner)) = guard.take_planner(reading) {
                drop(guard);
                // CPU work, outside the lock
                let result = planner.plan();
                let step = pool.lock().finish_planning(
                    unit,
                    result,
                    &mut self.metrics,
                    skip_errors,
                );
                return Self::step_result(step).unwrap_or(ScanAndReturn::Continue);
            }
        }

        // 4. active reader
        if self.yield_before_read && self.reader.is_some() {
            // Yield once before decoding, so the worker thread runs the I/O
            // tasks this stream just spawned first: Tokio places a task
            // spawned from a worker in that worker's LIFO slot, from which
            // other workers can't steal it, so the I/O would otherwise only
            // start once this stream yields, i.e. after decoding.
            self.yield_before_read = false;
            cx.waker().wake_by_ref();
            return ScanAndReturn::Return(Poll::Pending);
        }
        if let Some(ret) = self.poll_reader(cx) {
            return ret;
        }

        let mut guard = pool.lock();
        // 5. next morsel
        if let Some(morsel) = guard.take_morsel() {
            drop(guard);
            self.metrics.time_opening.stop();
            start_timer(&mut self.metrics.time_scanning_until_data);
            start_timer(&mut self.metrics.time_scanning_total);
            return match morsel.morsel.into_stream() {
                Ok(stream) => {
                    self.reader = Some(ActiveReader {
                        unit: morsel.unit,
                        bytes: morsel.bytes.unwrap_or(0),
                        stream,
                    });
                    ScanAndReturn::Continue
                }
                Err(err) => {
                    self.metrics.file_open_errors.add(1);
                    pool.lock().finish_read(
                        morsel.unit,
                        morsel.bytes.unwrap_or(0),
                        &mut self.metrics,
                    );
                    self.error(err)
                }
            };
        }

        // 6. open more work
        if !self.source_done && guard.may_open_unit() {
            match self.work_source.pop_front() {
                Some(file) => {
                    start_timer(&mut self.metrics.time_opening);
                    let unit = guard.begin_open();
                    drop(guard);
                    // CPU work, outside the lock
                    let planner = self.morselizer.plan_file(file);
                    let step = pool.lock().finish_open(
                        unit,
                        planner,
                        &mut self.metrics,
                        skip_errors,
                    );
                    return Self::step_result(step).unwrap_or(ScanAndReturn::Continue);
                }
                None => self.source_done = true,
            }
        }

        // Done once no file is left and nothing in the pool can be driven or
        // waited for. Morsels other streams are reading don't count: they
        // produce no further work.
        if self.source_done && !guard.has_pending_work() {
            return ScanAndReturn::Done(None);
        }
        // Something may have become ready since the checks above (e.g. while
        // planning outside the lock); otherwise wait to be woken by the pool
        // (completed I/O, new work, freed budget) or by in-flight inline I/O.
        if guard.has_ready_work(false) || (!self.source_done && guard.may_open_unit()) {
            return ScanAndReturn::Continue;
        }
        guard.register(self.waiter, cx.waker(), !self.source_done);
        self.yield_before_read = false;
        ScanAndReturn::Return(Poll::Pending)
    }

    /// Handle an error that ends the current item.
    fn error(&mut self, err: DataFusionError) -> ScanAndReturn {
        match self.on_error {
            OnError::Skip => ScanAndReturn::Continue,
            OnError::Fail => ScanAndReturn::Error(err),
        }
    }

    /// If there is an active reader, poll it for the next batch.
    fn poll_reader(&mut self, cx: &mut Context<'_>) -> Option<ScanAndReturn> {
        let reader = self.reader.as_mut()?;
        let next = match &mut reader.stream {
            MorselStream::Sync(iter) => Poll::Ready(iter.next()),
            MorselStream::Async(stream) => stream.poll_next_unpin(cx),
        };
        Some(match next {
            // Morsels should ideally only expose ready-to-decode streams,
            // but tolerate pending readers here.
            Poll::Pending => ScanAndReturn::Return(Poll::Pending),
            Poll::Ready(Some(Ok(batch))) => self.handle_batch(batch),
            Poll::Ready(Some(Err(err))) => {
                self.metrics.file_scan_errors.add(1);
                self.finish_reader();
                self.error(err)
            }
            Poll::Ready(None) => {
                self.finish_reader();
                ScanAndReturn::Continue
            }
        })
    }

    /// Drop the active reader and release its morsel.
    fn finish_reader(&mut self) {
        let reader = self.reader.take().expect("active reader");
        self.metrics.time_scanning_until_data.stop();
        self.metrics.time_scanning_total.stop();
        self.pool
            .lock()
            .finish_read(reader.unit, reader.bytes, &mut self.metrics);
    }

    fn handle_batch(&mut self, batch: RecordBatch) -> ScanAndReturn {
        self.metrics.time_scanning_until_data.stop();
        self.metrics.time_scanning_total.stop();
        // Apply any remaining row limit.
        let (batch, finished) = match &mut self.remain {
            Some(remain) => {
                if *remain > batch.num_rows() {
                    *remain -= batch.num_rows();
                    self.metrics.time_scanning_total.start();
                    (batch, false)
                } else {
                    let batch = batch.slice(0, *remain);
                    let open_files = self.pool.lock().open_files();
                    let done = open_files + self.work_source.skipped_on_limit();
                    self.metrics.files_processed.add(done);
                    // Its file was counted above: drop the reader without
                    // finishing it (scans with a limit have a private pool).
                    self.reader = None;
                    *remain = 0;
                    (batch, true)
                }
            }
            None => {
                self.metrics.time_scanning_total.start();
                (batch, false)
            }
        };
        if finished {
            ScanAndReturn::Done(Some(Ok(batch)))
        } else {
            ScanAndReturn::Return(Poll::Ready(Some(Ok(batch))))
        }
    }
}

impl Drop for ScanState {
    /// Release a morsel this stream was reading when it is dropped (finished,
    /// failed or cancelled), so that with a shared pool its bytes return to
    /// the budget and its unit of work can finish while siblings continue.
    /// A stream dropped while waiting stops waiting, so that wake-ups go to
    /// siblings instead.
    fn drop(&mut self) {
        if self.reader.is_some() {
            self.finish_reader();
        }
        self.pool.lock().unregister(self.waiter);
    }
}

/// What should be done on the next iteration of [`ScanState::poll_scan`]?
pub(super) enum ScanAndReturn {
    /// Poll again.
    Continue,
    /// Return the provided result without changing the outer state.
    Return(Poll<Option<Result<RecordBatch>>>),
    /// Update the outer `FileStreamState` to `Done` and return the provided result.
    Done(Option<Result<RecordBatch>>),
    /// Update the outer `FileStreamState` to `Error` and return the provided error.
    Error(DataFusionError),
}
