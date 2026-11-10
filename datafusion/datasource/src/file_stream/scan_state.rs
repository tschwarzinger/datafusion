// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::{BTreeMap, VecDeque};
use std::task::{Context, Poll};

use crate::morsel::{
    Morsel, MorselPlanner, MorselStream, Morselizer, PendingMorselPlanner,
};
use arrow::record_batch::RecordBatch;
use datafusion_common::{DataFusionError, Result};
use datafusion_common_runtime::SpawnedTask;
use datafusion_physical_plan::metrics::ScopedTimerGuard;
use futures::{FutureExt as _, StreamExt as _};

use super::work_source::{NextWork, WorkItem, WorkSource};
use super::{FileStreamMetrics, OnError, StartableTime};

/// Maximum number of units of work (files or donated planners) a single
/// stream keeps open at the same time, regardless of the I/O budget.
const MAX_OPEN_UNITS: usize = 64;

/// State [`FileStreamState::Scan`].
///
/// There is one `ScanState` per `FileStream`, and thus per output partition.
///
/// It groups together the lifecycle of scanning that partition's files:
/// unopened files, CPU-ready planners, pending planner I/O, ready morsels,
/// the active reader, and the metrics associated with processing that work.
///
/// # I/O budget
///
/// With an I/O budget of `0` (the default), the `ScanState` only ever has a
/// single I/O outstanding and does not plan new work while a reader is
/// active, i.e. I/O and decoding strictly alternate.
///
/// With a non-zero budget, the `ScanState` issues additional I/O (for the
/// next row groups of the current file, and for further files / donated
/// planners) while the current reader is decoding, as long as
///
/// ```text
/// bytes of in-flight I/O + bytes buffered in ready morsels and the reader <= budget
/// ```
///
/// At least one I/O is always allowed when nothing is in flight and no reader
/// is active, so a single item larger than the budget still makes progress.
/// I/O futures are lazy and are only polled (started) once they fit the
/// budget. Started I/O is spawned onto the Tokio runtime (when available, and only
/// with a non-zero budget) so
/// that it makes progress while this stream decodes.
///
/// # Ordering
///
/// When file order must be preserved, morsels are only read from the oldest
/// open unit of work, in the order they were produced, so prefetching never
/// reorders output. Otherwise any ready morsel may be read, and planners that
/// split a file into several independent planners may donate the extra
/// planners to sibling streams via the shared work source.
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
///       +-------- poll until ready
/// ```
///
/// [`FileStreamState::Scan`]: super::FileStreamState::Scan
pub(super) struct ScanState {
    /// Unopened files that still need to be planned for this stream.
    work_source: WorkSource,
    /// Remaining row limit, if any.
    remain: Option<usize>,
    /// The morselizer used to plan files.
    morselizer: Box<dyn Morselizer>,
    /// Behavior if opening or scanning a file fails.
    on_error: OnError,
    /// Maximum bytes of in-flight + buffered I/O (see type docs).
    io_budget: usize,
    /// In which order must / should morsels be read?
    read_order: ReadOrder,
    /// May planners split files and donate parts to siblings?
    may_donate: bool,
    /// Open units of work, keyed by an increasing id.
    units: BTreeMap<u64, Unit>,
    /// Id for the next unit of work.
    next_unit: u64,
    /// CPU-ready planners.
    ready_planners: VecDeque<(u64, Box<dyn MorselPlanner>)>,
    /// Planner I/O that has not been started (polled) yet.
    unstarted_io: VecDeque<(u64, PendingMorselPlanner)>,
    /// Planner I/O that has been started.
    in_flight: Vec<InFlightIo>,
    /// Ready morsels in production order.
    ready_morsels: VecDeque<ReadyMorsel>,
    /// The active reader, if any.
    reader: Option<ActiveReader>,
    /// Sum of the known sizes of in-flight I/O.
    in_flight_bytes: usize,
    /// Number of in-flight I/Os of unknown size (e.g. metadata reads).
    in_flight_unknown: usize,
    /// Sum of `buffered_bytes` of ready morsels and the active reader.
    buffered_bytes: usize,
    /// Set once the work source returned `Done`.
    source_done: bool,
    /// Metrics for the active scan queues.
    metrics: FileStreamMetrics,
}

/// Book keeping for one unit of work (a file, or a donated planner).
struct Unit {
    /// Number of planners, I/Os, morsels and readers alive for this unit.
    outstanding: usize,
    /// Whether finishing this unit counts as a processed file.
    counts_file: bool,
    /// Whether this unit was taken from a shared queue as a file and has not
    /// yet settled (see `SharedWorkSource::settle`).
    unsettled: bool,
}

struct ReadyMorsel {
    unit: u64,
    morsel: Box<dyn Morsel>,
    bytes: usize,
}

struct ActiveReader {
    unit: u64,
    bytes: usize,
    stream: MorselStream,
}

struct InFlightIo {
    unit: u64,
    bytes: Option<usize>,
    future: IoFuture,
}

enum IoFuture {
    /// Polled inline on the stream's task (no Tokio runtime available).
    Inline(PendingMorselPlanner),
    /// Spawned onto the Tokio runtime.
    Spawned(SpawnedTask<Result<Box<dyn MorselPlanner>>>),
}

impl IoFuture {
    fn start(pending: PendingMorselPlanner, spawn: bool) -> Self {
        if spawn && tokio::runtime::Handle::try_current().is_ok() {
            Self::Spawned(SpawnedTask::spawn(pending.into_future()))
        } else {
            Self::Inline(pending)
        }
    }

    fn poll(&mut self, cx: &mut Context<'_>) -> Poll<Result<Box<dyn MorselPlanner>>> {
        match self {
            Self::Inline(pending) => pending.poll_unpin(cx),
            Self::Spawned(task) => task.poll_unpin(cx).map(|res| {
                res.map_err(|e| DataFusionError::External(Box::new(e)))
                    .and_then(|r| r)
            }),
        }
    }
}

fn start_timer(time: &mut StartableTime) {
    if time.start.is_none() {
        time.start();
    }
}

impl ScanState {
    pub(super) fn new(
        work_source: WorkSource,
        remain: Option<usize>,
        morselizer: Box<dyn Morselizer>,
        on_error: OnError,
        metrics: FileStreamMetrics,
    ) -> Self {
        Self {
            work_source,
            remain,
            morselizer,
            on_error,
            io_budget: 0,
            read_order: ReadOrder::Required,
            may_donate: false,
            units: BTreeMap::new(),
            next_unit: 0,
            ready_planners: Default::default(),
            unstarted_io: Default::default(),
            in_flight: Default::default(),
            ready_morsels: Default::default(),
            reader: None,
            in_flight_bytes: 0,
            in_flight_unknown: 0,
            buffered_bytes: 0,
            source_done: false,
            metrics,
        }
    }

    /// Configure the I/O budget, in bytes. `0` means a single outstanding I/O.
    ///
    /// The budget is ignored (treated as `0`) when the scan has a row limit,
    /// to avoid reading ahead data that will not be needed.
    pub(super) fn with_io_budget(mut self, io_budget: usize) -> Self {
        self.io_budget = if self.remain.is_some() { 0 } else { io_budget };
        self
    }

    /// Configure the order in which morsels are read.
    pub(super) fn with_read_order(mut self, read_order: ReadOrder) -> Self {
        self.read_order = read_order;
        self
    }

    /// Configure whether planners may split files whose parts are donated to
    /// sibling streams. Only then do idle siblings wait for files that are
    /// still being planned by another stream.
    pub(super) fn with_may_donate(mut self, may_donate: bool) -> Self {
        self.may_donate = may_donate;
        self
    }

    /// Updates how scan errors are handled while the stream is still active.
    pub(super) fn set_on_error(&mut self, on_error: OnError) {
        self.on_error = on_error;
    }

    fn legacy(&self) -> bool {
        self.io_budget == 0
    }

    fn reserved_bytes(&self) -> usize {
        self.in_flight_bytes + self.buffered_bytes
    }

    // ---------------------------------------------------------------
    // Unit book keeping
    // ---------------------------------------------------------------

    fn new_unit(&mut self, counts_file: bool, unsettled: bool) -> u64 {
        let id = self.next_unit;
        self.next_unit += 1;
        self.units.insert(
            id,
            Unit {
                outstanding: 0,
                counts_file,
                unsettled,
            },
        );
        id
    }

    fn add_outstanding(&mut self, unit: u64, n: usize) {
        if let Some(u) = self.units.get_mut(&unit) {
            u.outstanding += n;
        }
    }

    /// Decrement the outstanding count of `unit`, finishing it at zero.
    fn release(&mut self, unit: u64) {
        let finished = match self.units.get_mut(&unit) {
            Some(u) => {
                u.outstanding = u.outstanding.saturating_sub(1);
                u.outstanding == 0
            }
            None => false,
        };
        if finished {
            let u = self.units.remove(&unit).expect("unit present");
            if u.counts_file {
                self.metrics.files_processed.add(1);
            }
            if u.unsettled {
                self.settle_shared();
            }
            if self.units.is_empty() {
                self.metrics.time_opening.stop();
            }
        }
    }

    fn settle(&mut self, unit: u64) {
        let settle = match self.units.get_mut(&unit) {
            Some(u) if u.unsettled => {
                u.unsettled = false;
                true
            }
            _ => false,
        };
        if settle {
            self.settle_shared();
        }
    }

    fn settle_shared(&self) {
        if let Some(shared) = self.work_source.shared() {
            shared.settle();
        }
    }

    /// Handle an error for `unit` according to `on_error`.
    fn handle_error(&mut self, unit: u64, err: DataFusionError) -> ScanAndReturn {
        match self.on_error {
            OnError::Skip => {
                self.release(unit);
                ScanAndReturn::Continue
            }
            OnError::Fail => ScanAndReturn::Error(err),
        }
    }

    // ---------------------------------------------------------------
    // Main loop
    // ---------------------------------------------------------------

    /// Drives one iteration of the active scan state.
    ///
    /// Work is attempted in this order:
    /// 1. resolve any started planner I/O
    /// 2. start planner I/O that fits within the I/O budget
    /// 3. run CPU planning on ready planners
    /// 4. poll the active reader
    /// 5. turn a ready morsel into the active reader
    /// 6. open the next unit of work (file or donated planner)
    ///
    /// The return [`ScanAndReturn`] tells `poll_inner` how to update the
    /// outer `FileStreamState`.
    pub(super) fn poll_scan(&mut self, cx: &mut Context<'_>) -> ScanAndReturn {
        let time_processing = self.metrics.time_processing.clone();
        let _timer: ScopedTimerGuard<'_> = time_processing.timer();

        // 1. resolve started I/O
        if let Some(ret) = self.poll_in_flight(cx) {
            return ret;
        }

        // 2. start I/O within budget
        if self.start_io() {
            return ScanAndReturn::Continue;
        }

        // 3. CPU planning
        if let Some(ret) = self.plan_ready() {
            return ret;
        }

        // 4. active reader
        if let Some(ret) = self.poll_reader(cx) {
            return ret;
        }

        // 5. next morsel
        if let Some(ret) = self.start_next_morsel() {
            return ret;
        }

        // 6. open more work
        if self.may_open_unit() {
            match self.work_source.poll_next(cx.waker(), self.may_donate) {
                NextWork::Item(item) => return self.open_unit(item),
                NextWork::Wait => {}
                NextWork::Done => self.source_done = true,
            }
        }

        if self.units.is_empty() && self.source_done {
            return ScanAndReturn::Done(None);
        }
        // Wakers are registered by in-flight I/O, async readers or the
        // shared work source.
        ScanAndReturn::Return(Poll::Pending)
    }

    /// Poll all started I/O. Returns `Some` on error.
    fn poll_in_flight(&mut self, cx: &mut Context<'_>) -> Option<ScanAndReturn> {
        let mut i = 0;
        while i < self.in_flight.len() {
            match self.in_flight[i].future.poll(cx) {
                Poll::Pending => i += 1,
                Poll::Ready(res) => {
                    let io = self.in_flight.swap_remove(i);
                    match io.bytes {
                        Some(b) => self.in_flight_bytes -= b,
                        None => self.in_flight_unknown -= 1,
                    }
                    match res {
                        Ok(planner) => {
                            // the planner replaces the I/O as outstanding item
                            self.ready_planners.push_back((io.unit, planner));
                        }
                        Err(err) => {
                            self.metrics.file_open_errors.add(1);
                            let ret = self.handle_error(io.unit, err);
                            if !matches!(ret, ScanAndReturn::Continue) {
                                return Some(ret);
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Start unstarted I/O as allowed by the budget. Returns true if any I/O
    /// was started (it must be polled to register a waker).
    fn start_io(&mut self) -> bool {
        let mut started = false;
        while let Some((_, pending)) = self.unstarted_io.front() {
            let bytes = pending.io_bytes();
            // Legacy mode: a single outstanding I/O, which may overlap the
            // active reader. Budgeted mode: always allow one I/O when nothing
            // else is in flight or being decoded.
            let nothing_running = self.in_flight.is_empty()
                && (self.legacy()
                    || (self.reader.is_none() && !self.has_eligible_morsel()));
            let fits = !self.legacy()
                && self.reserved_bytes() + bytes.unwrap_or(0) <= self.io_budget;
            if !(nothing_running || fits) {
                break;
            }
            let (unit, pending) = self.unstarted_io.pop_front().expect("front");
            match bytes {
                Some(b) => self.in_flight_bytes += b,
                None => self.in_flight_unknown += 1,
            }
            self.in_flight.push(InFlightIo {
                unit,
                bytes,
                future: IoFuture::start(pending, !self.legacy()),
            });
            started = true;
        }
        started
    }

    fn has_eligible_morsel(&self) -> bool {
        self.eligible_morsel_index().is_some()
    }

    /// Index of the next morsel that may become the active reader.
    fn eligible_morsel_index(&self) -> Option<usize> {
        if self.ready_morsels.is_empty() {
            return None;
        }
        if self.read_order == ReadOrder::Any {
            return Some(0);
        }
        // Only the oldest open unit may produce output; its morsels are in
        // production (= logical) order. Units are opened in the order the
        // work source and morselizer prioritized the work (split planners
        // get a unit each), so this reads in planned order even though their
        // I/O may complete in any order.
        let oldest = *self.units.keys().next()?;
        self.ready_morsels.iter().position(|m| m.unit == oldest)
    }

    /// Plan ready planners (all of them, or one in legacy mode).
    fn plan_ready(&mut self) -> Option<ScanAndReturn> {
        let mut planned = false;
        loop {
            if self.legacy()
                && (self.reader.is_some()
                    || !self.ready_morsels.is_empty()
                    || !self.in_flight.is_empty()
                    || !self.unstarted_io.is_empty())
            {
                break;
            }
            let Some((unit, planner)) = self.ready_planners.pop_front() else {
                break;
            };
            planned = true;
            match planner.plan() {
                Ok(Some(mut plan)) => {
                    let morsels = plan.take_morsels();
                    let mut planners = plan.take_ready_planners();
                    let pending = plan.take_pending_planner();

                    if !morsels.is_empty() {
                        self.settle(unit);
                    }
                    // A planner that split its work into several independent
                    // planners (only done when the order is not required):
                    // keep the first, donate the others to sibling streams if
                    // possible, otherwise track each as its own unit so they
                    // are read in planned order (see `eligible_morsel_index`).
                    if planners.len() > 1 && self.read_order != ReadOrder::Required {
                        let split = planners.split_off(1);
                        match self.work_source.shared() {
                            Some(shared) if self.may_donate => shared.donate(split),
                            _ => {
                                for planner in split {
                                    let child = self.new_unit(false, false);
                                    self.add_outstanding(child, 1);
                                    self.ready_planners.push_back((child, planner));
                                }
                            }
                        }
                        self.settle(unit);
                    }

                    self.add_outstanding(
                        unit,
                        morsels.len() + planners.len() + usize::from(pending.is_some()),
                    );
                    for morsel in morsels {
                        let bytes = morsel.buffered_bytes();
                        self.buffered_bytes += bytes;
                        self.ready_morsels.push_back(ReadyMorsel {
                            unit,
                            morsel,
                            bytes,
                        });
                    }
                    for planner in planners {
                        self.ready_planners.push_back((unit, planner));
                    }
                    if let Some(pending) = pending {
                        self.unstarted_io.push_back((unit, pending));
                    }
                    // the consumed planner
                    self.release(unit);
                }
                Ok(None) => self.release(unit),
                Err(err) => {
                    self.metrics.file_open_errors.add(1);
                    let ret = self.handle_error(unit, err);
                    if !matches!(ret, ScanAndReturn::Continue) {
                        return Some(ret);
                    }
                }
            }
            if self.legacy() {
                break;
            }
        }
        planned.then_some(ScanAndReturn::Continue)
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
                let unit = self.finish_reader();
                self.handle_error(unit, err)
            }
            Poll::Ready(None) => {
                let unit = self.finish_reader();
                self.release(unit);
                ScanAndReturn::Continue
            }
        })
    }

    /// Drop the active reader, returning its unit.
    fn finish_reader(&mut self) -> u64 {
        let reader = self.reader.take().expect("active reader");
        self.buffered_bytes -= reader.bytes;
        self.metrics.time_scanning_until_data.stop();
        self.metrics.time_scanning_total.stop();
        reader.unit
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
                    let open_files =
                        self.units.values().filter(|u| u.counts_file).count();
                    let done = open_files + self.work_source.skipped_on_limit();
                    self.metrics.files_processed.add(done);
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

    /// Turn the next eligible ready morsel into the active reader.
    fn start_next_morsel(&mut self) -> Option<ScanAndReturn> {
        debug_assert!(self.reader.is_none());
        let idx = self.eligible_morsel_index()?;
        let ReadyMorsel {
            unit,
            morsel,
            bytes,
        } = self.ready_morsels.remove(idx).expect("index valid");
        self.metrics.time_opening.stop();
        start_timer(&mut self.metrics.time_scanning_until_data);
        start_timer(&mut self.metrics.time_scanning_total);
        match morsel.into_stream() {
            Ok(stream) => {
                self.reader = Some(ActiveReader {
                    unit,
                    bytes,
                    stream,
                });
                Some(ScanAndReturn::Continue)
            }
            Err(err) => {
                self.buffered_bytes -= bytes;
                self.metrics.file_open_errors.add(1);
                Some(self.handle_error(unit, err))
            }
        }
    }

    /// May this stream open another unit of work?
    fn may_open_unit(&self) -> bool {
        if self.source_done {
            return false;
        }
        if self.units.is_empty() {
            return true;
        }
        !self.legacy()
            && self.units.len() < MAX_OPEN_UNITS
            // budget is not already saturated
            && self.unstarted_io.is_empty()
            && self.ready_planners.is_empty()
            // don't open files in bursts while their I/O size is unknown
            && self.in_flight_unknown == 0
            // morsels that do not report their size (e.g. the `FileOpener`
            // adapter) can't be charged against the budget: don't open more
            // work while such a morsel is waiting
            && self.ready_morsels.iter().all(|m| m.bytes > 0)
            && self.reserved_bytes() < self.io_budget
    }

    fn open_unit(&mut self, item: WorkItem) -> ScanAndReturn {
        start_timer(&mut self.metrics.time_opening);
        match item {
            WorkItem::Planner(planner) => {
                let unit = self.new_unit(false, false);
                self.add_outstanding(unit, 1);
                self.ready_planners.push_back((unit, planner));
                ScanAndReturn::Continue
            }
            WorkItem::File(part_file) => {
                let unsettled = self.work_source.shared().is_some() && self.may_donate;
                let unit = self.new_unit(true, unsettled);
                match self.morselizer.plan_file(*part_file) {
                    Ok(planner) => {
                        self.metrics.files_opened.add(1);
                        self.add_outstanding(unit, 1);
                        self.ready_planners.push_back((unit, planner));
                        ScanAndReturn::Continue
                    }
                    Err(err) => {
                        self.metrics.file_open_errors.add(1);
                        // finish the (empty) unit
                        self.add_outstanding(unit, 1);
                        self.handle_error(unit, err)
                    }
                }
            }
        }
    }
}

/// In which order a [`ScanState`] reads the morsels it has ready.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ReadOrder {
    /// The output order is part of the scan's contract (e.g. the scan has an
    /// output ordering): morsels are read strictly in planned order and work
    /// is never split or shared.
    Required,
    /// Any order is correct, but the planned order is a priority that makes
    /// the scan cheaper (e.g. row groups reordered so a TopK dynamic filter
    /// tightens early). Morsels are read in planned order, waiting for the
    /// next one if needed, while I/O is still prefetched and work may still
    /// be split and shared with siblings.
    Preferred,
    /// Read whatever morsel is ready first.
    Any,
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
