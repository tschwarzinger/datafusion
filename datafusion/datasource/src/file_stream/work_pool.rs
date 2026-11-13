// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`WorkPool`]: all outstanding work of a scan that has been opened but not
//! yet read, driven by one stream or shared by sibling streams.

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Weak};
use std::task::{Context, Poll, Waker};

use datafusion_common::{DataFusionError, Result};
use datafusion_common_runtime::SpawnedTask;
use futures::FutureExt as _;
use parking_lot::Mutex;

use crate::morsel::{Morsel, MorselPlan, MorselPlanner, PendingMorselPlanner};
use datafusion_execution::IoBudget;

use super::FileStreamMetrics;

/// Maximum number of units of work open per stream sharing a pool,
/// regardless of the I/O budget.
const MAX_OPEN_UNITS_PER_STREAM: usize = 64;

/// A [`WorkPool`] that may be shared by sibling streams.
pub(super) type SharedPool = Arc<Mutex<WorkPool>>;

/// In which order morsels are read.
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

/// Configuration of a [`WorkPool`].
#[derive(Debug, Clone)]
pub(super) struct PoolConfig {
    /// Query-wide budget for in-flight I/O plus fetched-but-not-yet-read
    /// morsels. `None` (or a disabled budget) means a single outstanding I/O
    /// at a time (legacy mode).
    pub(super) io_budget: Option<Arc<IoBudget>>,
    pub(super) read_order: ReadOrder,
    /// Number of streams driving this pool.
    pub(super) streams: usize,
}

/// All outstanding work of a scan: units of work that were opened (files, or
/// independently readable parts of files), their CPU-ready planners, their
/// I/O (not started, in flight, or completed but not yet collected), and the
/// morsels fetched for them.
///
/// Work stays in the pool until it is done. A stream takes an item out only
/// for the CPU step it performs within its current poll (planning a planner,
/// reading a morsel), so when the pool is shared between sibling streams,
/// **any** stream can drive **any** outstanding item:
///
/// * Started I/O runs as a spawned task that delivers its result back into
///   the pool and wakes all waiting streams, so it completes without any
///   particular stream being polled.
/// * A stream only finishes once the pool has nothing left it could drive and
///   no unopened files remain, rather than as soon as the queue of unopened
///   files is empty. Waiting is safe: everything it waits for makes progress
///   on its own (spawned I/O) or within another stream's current poll
///   (planning).
///
/// Only items that are being *read* (a stream's active reader) belong to a
/// stream; finishing does not wait for those.
pub(super) struct WorkPool {
    config: PoolConfig,
    /// Open units of work, keyed by an increasing id (= planned order).
    units: BTreeMap<u64, Unit>,
    next_unit: u64,
    next_io: u64,
    /// CPU-ready planners.
    ready_planners: VecDeque<(u64, Box<dyn MorselPlanner>)>,
    /// Planners (or files) a stream is currently planning, outside the lock.
    planning: usize,
    /// Planner I/O that has not been started (polled) yet.
    unstarted_io: VecDeque<(u64, PendingMorselPlanner)>,
    /// Planner I/O that has been started.
    in_flight: Vec<InFlightIo>,
    /// Results of spawned I/O, not yet collected.
    completed: Vec<(u64, Result<Box<dyn MorselPlanner>>)>,
    /// Fetched morsels in production order.
    ready_morsels: VecDeque<ReadyMorsel>,
    /// Number of morsels being read by some stream.
    active_readers: usize,
    /// Sum of the known sizes of in-flight I/O.
    in_flight_bytes: usize,
    /// Number of in-flight I/Os of unknown size (e.g. metadata reads).
    in_flight_unknown: usize,
    /// Sum of `buffered_bytes` of ready morsels and morsels being read.
    buffered_bytes: usize,
    /// Streams waiting for work.
    wakers: Vec<Waker>,
}

/// Book keeping for one unit of work.
struct Unit {
    /// Number of planners, I/Os and morsels (ready or being read) alive for
    /// this unit.
    outstanding: usize,
    /// Whether finishing this unit counts as a processed file.
    counts_file: bool,
    /// Number of this unit's morsels being read.
    reading: usize,
}

/// A morsel ready to be read.
pub(super) struct ReadyMorsel {
    pub(super) unit: u64,
    pub(super) morsel: Box<dyn Morsel>,
    pub(super) bytes: usize,
}

struct InFlightIo {
    id: u64,
    unit: u64,
    bytes: Option<usize>,
    future: IoFuture,
}

enum IoFuture {
    /// Polled inline by whichever stream drives the pool (legacy mode, or no
    /// Tokio runtime available).
    Inline(PendingMorselPlanner),
    /// Spawned onto the Tokio runtime; delivers its result into `completed`.
    /// Kept so the task is aborted if the pool is dropped.
    Spawned(#[expect(dead_code)] SpawnedTask<()>),
}

/// What a stream should do after an operation on the pool.
pub(super) enum PoolStep {
    /// Something happened; poll again.
    Progress,
    /// Nothing to do.
    Idle,
    /// An error to return to the consumer.
    Error(DataFusionError),
}

impl WorkPool {
    pub(super) fn new(config: PoolConfig) -> Self {
        Self {
            config,
            units: BTreeMap::new(),
            next_unit: 0,
            next_io: 0,
            ready_planners: VecDeque::new(),
            planning: 0,
            unstarted_io: VecDeque::new(),
            in_flight: Vec::new(),
            completed: Vec::new(),
            ready_morsels: VecDeque::new(),
            active_readers: 0,
            in_flight_bytes: 0,
            in_flight_unknown: 0,
            buffered_bytes: 0,
            wakers: Vec::new(),
        }
    }

    pub(super) fn new_shared(config: PoolConfig) -> SharedPool {
        Arc::new(Mutex::new(Self::new(config)))
    }

    fn legacy(&self) -> bool {
        self.budget().is_none()
    }

    /// The query-wide budget, if prefetching is enabled.
    fn budget(&self) -> Option<&Arc<IoBudget>> {
        self.config.io_budget.as_ref().filter(|b| b.is_enabled())
    }

    // ---------------------------------------------------------------
    // Units
    // ---------------------------------------------------------------

    fn new_unit(&mut self, counts_file: bool) -> u64 {
        let id = self.next_unit;
        self.next_unit += 1;
        self.units.insert(
            id,
            Unit {
                outstanding: 0,
                counts_file,
                reading: 0,
            },
        );
        id
    }

    fn add_outstanding(&mut self, unit: u64, n: usize) {
        if let Some(u) = self.units.get_mut(&unit) {
            u.outstanding += n;
        }
    }

    /// Release one outstanding item of `unit`, finishing it at zero.
    pub(super) fn release(&mut self, unit: u64, metrics: &mut FileStreamMetrics) {
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
                metrics.files_processed.add(1);
            }
            if self.units.is_empty() {
                metrics.time_opening.stop();
            }
            self.wake_all();
        }
    }

    /// Number of open units that count as files (for limit accounting).
    pub(super) fn open_files(&self) -> usize {
        self.units.values().filter(|u| u.counts_file).count()
    }

    // ---------------------------------------------------------------
    // Waking
    // ---------------------------------------------------------------

    fn wake_all(&mut self) {
        for waker in self.wakers.drain(..) {
            waker.wake();
        }
    }

    /// Register a stream that found nothing to do.
    ///
    /// With `may_open_more`, the stream could open more work once budget is
    /// available, so it also waits for budget released by other scans of the
    /// query.
    pub(super) fn register(&mut self, waker: &Waker, may_open_more: bool) {
        if !self.wakers.iter().any(|w| w.will_wake(waker)) {
            self.wakers.push(waker.clone());
        }
        if (may_open_more || !self.unstarted_io.is_empty())
            && let Some(budget) = self.budget()
        {
            budget.register(waker);
        }
    }

    // ---------------------------------------------------------------
    // I/O
    // ---------------------------------------------------------------

    /// Collect completed I/O: poll inline I/O and take results delivered by
    /// spawned I/O. Completed I/O is replaced by its CPU-ready planner.
    pub(super) fn collect_io(
        &mut self,
        cx: &mut Context<'_>,
        metrics: &mut FileStreamMetrics,
        skip_errors: bool,
    ) -> PoolStep {
        let mut results = std::mem::take(&mut self.completed);
        let mut i = 0;
        while i < self.in_flight.len() {
            if let IoFuture::Inline(future) = &mut self.in_flight[i].future
                && let Poll::Ready(result) = future.poll_unpin(cx)
            {
                results.push((self.in_flight[i].id, result));
            }
            i += 1;
        }
        let mut step = PoolStep::Idle;
        for (id, result) in results {
            let Some(idx) = self.in_flight.iter().position(|io| io.id == id) else {
                continue;
            };
            let io = self.in_flight.swap_remove(idx);
            match io.bytes {
                Some(b) => {
                    self.in_flight_bytes -= b;
                    if let Some(budget) = self.budget() {
                        budget.release(b);
                    }
                }
                None => self.in_flight_unknown -= 1,
            }
            step = PoolStep::Progress;
            match result {
                // the planner replaces the I/O as outstanding item
                Ok(planner) => self.ready_planners.push_back((io.unit, planner)),
                Err(err) => {
                    metrics.file_open_errors.add(1);
                    if skip_errors {
                        self.release(io.unit, metrics);
                    } else {
                        return PoolStep::Error(err);
                    }
                }
            }
        }
        step
    }

    /// May the I/O at the front of `unstarted_io` be started now?
    ///
    /// With `reserve`, the bytes are reserved from the query-wide budget if
    /// the I/O may start.
    fn admit_io(&self, stream_reading: bool, reserve: bool) -> bool {
        let Some((_, pending)) = self.unstarted_io.front() else {
            return false;
        };
        let Some(budget) = self.budget() else {
            // a single outstanding I/O, which may overlap the active reader
            return self.in_flight.is_empty();
        };
        let bytes = pending.io_bytes().unwrap_or(0);
        // Always allow one I/O when this scan has nothing in flight or being
        // read, so a scan makes progress even if other scans of the query hold
        // the whole budget, and an item larger than the budget still runs.
        let nothing_running = self.in_flight.is_empty()
            && self.active_readers == 0
            && !stream_reading
            && self.eligible_morsel_index().is_none();
        if nothing_running {
            if reserve {
                budget.force_reserve(bytes);
            }
            return true;
        }
        if reserve {
            budget.try_reserve(bytes)
        } else {
            budget.reserved() + bytes <= budget.limit()
        }
    }

    fn can_start_io(&self, stream_reading: bool) -> bool {
        self.admit_io(stream_reading, false)
    }

    /// Start I/O as allowed by the budget. Returns true if any I/O was started.
    pub(super) fn start_io(&mut self, this: &SharedPool, stream_reading: bool) -> bool {
        let mut started = false;
        while self.admit_io(stream_reading, true) {
            let (unit, pending) = self.unstarted_io.pop_front().expect("front");
            let bytes = pending.io_bytes();
            match bytes {
                Some(b) => self.in_flight_bytes += b,
                None => self.in_flight_unknown += 1,
            }
            let id = self.next_io;
            self.next_io += 1;
            let future =
                if !self.legacy() && tokio::runtime::Handle::try_current().is_ok() {
                    IoFuture::Spawned(spawn_delivering(Arc::downgrade(this), id, pending))
                } else {
                    IoFuture::Inline(pending)
                };
            self.in_flight.push(InFlightIo {
                id,
                unit,
                bytes,
                future,
            });
            started = true;
        }
        started
    }

    // ---------------------------------------------------------------
    // Planning
    // ---------------------------------------------------------------

    /// Is a planner ready and may it be planned now?
    fn can_plan(&self, stream_reading: bool) -> bool {
        // Legacy mode: plan only when there is nothing else to do, so I/O and
        // decoding strictly alternate.
        !self.ready_planners.is_empty()
            && !(self.legacy()
                && (stream_reading
                    || !self.ready_morsels.is_empty()
                    || !self.in_flight.is_empty()
                    || !self.unstarted_io.is_empty()))
    }

    /// Take a planner to plan outside the lock, if planning is allowed now.
    ///
    /// The caller must hand the result to [`Self::finish_planning`].
    pub(super) fn take_planner(
        &mut self,
        stream_reading: bool,
    ) -> Option<(u64, Box<dyn MorselPlanner>)> {
        if !self.can_plan(stream_reading) {
            return None;
        }
        let planner = self.ready_planners.pop_front()?;
        self.planning += 1;
        Some(planner)
    }

    /// Apply the result of planning a planner of `unit` taken with
    /// [`Self::take_planner`].
    pub(super) fn finish_planning(
        &mut self,
        unit: u64,
        result: Result<Option<MorselPlan>>,
        metrics: &mut FileStreamMetrics,
        skip_errors: bool,
    ) -> PoolStep {
        self.planning -= 1;
        // others may be waiting for planning to finish
        self.wake_all();
        let mut plan = match result {
            Ok(Some(plan)) => plan,
            Ok(None) => {
                self.release(unit, metrics);
                return PoolStep::Progress;
            }
            Err(err) => {
                metrics.file_open_errors.add(1);
                if skip_errors {
                    self.release(unit, metrics);
                    return PoolStep::Progress;
                }
                return PoolStep::Error(err);
            }
        };
        let morsels = plan.take_morsels();
        let mut planners = plan.take_ready_planners();
        let pending = plan.take_pending_planner();

        // A planner that split its work into several independent planners
        // (only done when the order is not required). When the planned order
        // is preferred, give each part a unit of its own so "planned order"
        // is well defined (see `eligible_morsel_index`).
        if planners.len() > 1 && self.config.read_order == ReadOrder::Preferred {
            for planner in planners.split_off(1) {
                let child = self.new_unit(false);
                self.add_outstanding(child, 1);
                self.ready_planners.push_back((child, planner));
            }
        }

        self.add_outstanding(
            unit,
            morsels.len() + planners.len() + usize::from(pending.is_some()),
        );
        for morsel in morsels {
            let bytes = morsel.buffered_bytes();
            self.buffered_bytes += bytes;
            // already fetched: charged to the budget until read
            if let Some(budget) = self.budget() {
                budget.force_reserve(bytes);
            }
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
        self.release(unit, metrics);
        PoolStep::Progress
    }

    // ---------------------------------------------------------------
    // Opening work
    // ---------------------------------------------------------------

    /// May another unit of work (file) be opened?
    pub(super) fn may_open_unit(&self) -> bool {
        if self.units.is_empty() && self.planning == 0 {
            return true;
        }
        let streams = self.config.streams.max(1);
        !self.legacy()
            && self.units.len() < MAX_OPEN_UNITS_PER_STREAM * streams
            // budget is not already saturated
            && self.unstarted_io.is_empty()
            && self.ready_planners.is_empty()
            // don't open files in bursts while their I/O size is unknown:
            // at most one such file (being planned or loading metadata) per
            // stream
            && self.in_flight_unknown + self.planning < streams
            // morsels that do not report their size (e.g. the `FileOpener`
            // adapter) can't be charged against the budget: don't open more
            // work while such a morsel is waiting
            && self.ready_morsels.iter().all(|m| m.bytes > 0)
            && self.budget().is_some_and(|b| b.has_available())
    }

    /// Register a newly opened file whose planner is being created outside
    /// the lock. Returns its unit; finish with [`Self::finish_open`].
    pub(super) fn begin_open(&mut self) -> u64 {
        let unit = self.new_unit(true);
        self.add_outstanding(unit, 1);
        self.planning += 1;
        unit
    }

    /// Apply the result of creating the planner for `unit`.
    pub(super) fn finish_open(
        &mut self,
        unit: u64,
        planner: Result<Box<dyn MorselPlanner>>,
        metrics: &mut FileStreamMetrics,
        skip_errors: bool,
    ) -> PoolStep {
        self.planning -= 1;
        self.wake_all();
        match planner {
            Ok(planner) => {
                metrics.files_opened.add(1);
                self.ready_planners.push_back((unit, planner));
                PoolStep::Progress
            }
            Err(err) => {
                metrics.file_open_errors.add(1);
                if skip_errors {
                    self.release(unit, metrics);
                    PoolStep::Progress
                } else {
                    PoolStep::Error(err)
                }
            }
        }
    }

    // ---------------------------------------------------------------
    // Reading
    // ---------------------------------------------------------------

    /// Index of the next morsel that may be read.
    fn eligible_morsel_index(&self) -> Option<usize> {
        if self.ready_morsels.is_empty() {
            return None;
        }
        if self.config.read_order == ReadOrder::Any {
            return Some(0);
        }
        // Read the oldest unit that no stream is reading yet; its morsels are
        // in production (= logical) order. Units are opened in the order the
        // work source and morselizer prioritized the work, so this reads in
        // planned order even though I/O may complete in any order.
        let (&next, _) = self.units.iter().find(|(_, u)| u.reading == 0)?;
        self.ready_morsels.iter().position(|m| m.unit == next)
    }

    /// Take the next morsel to read. Finish with [`Self::finish_read`].
    pub(super) fn take_morsel(&mut self) -> Option<ReadyMorsel> {
        let idx = self.eligible_morsel_index()?;
        let morsel = self.ready_morsels.remove(idx).expect("index valid");
        if let Some(u) = self.units.get_mut(&morsel.unit) {
            u.reading += 1;
        }
        self.active_readers += 1;
        Some(morsel)
    }

    /// A morsel taken with [`Self::take_morsel`] has been read (or failed).
    pub(super) fn finish_read(
        &mut self,
        unit: u64,
        bytes: usize,
        metrics: &mut FileStreamMetrics,
    ) {
        self.buffered_bytes -= bytes;
        if let Some(budget) = self.budget() {
            budget.release(bytes);
        }
        self.active_readers -= 1;
        if let Some(u) = self.units.get_mut(&unit) {
            u.reading -= 1;
        }
        self.release(unit, metrics);
        // budget was freed, and the next unit may now be eligible
        self.wake_all();
    }

    // ---------------------------------------------------------------
    // Finishing
    // ---------------------------------------------------------------

    /// Is there anything left that a stream (without an active reader) could
    /// drive or wait for? Morsels being read by other streams do not count.
    pub(super) fn has_pending_work(&self) -> bool {
        self.planning > 0
            || !self.ready_planners.is_empty()
            || !self.unstarted_io.is_empty()
            || !self.in_flight.is_empty()
            || !self.completed.is_empty()
            || !self.ready_morsels.is_empty()
    }

    /// Could a stream make progress right now? Used to avoid missing a
    /// wake-up between dropping the lock and registering the waker.
    pub(super) fn has_ready_work(&self, stream_reading: bool) -> bool {
        !self.completed.is_empty()
            || self.can_plan(stream_reading)
            || self.can_start_io(stream_reading)
            || (!stream_reading && self.eligible_morsel_index().is_some())
    }
}

impl Drop for WorkPool {
    /// Return this scan's reservations to the query-wide budget, e.g. when
    /// the scan is dropped early (LIMIT, cancellation).
    fn drop(&mut self) {
        if let Some(budget) = self.budget() {
            budget.release(self.in_flight_bytes + self.buffered_bytes);
        }
    }
}

/// Spawn `pending` onto the Tokio runtime. On completion its result is
/// delivered into the pool and all waiting streams are woken, so the I/O
/// completes without any particular stream being polled.
fn spawn_delivering(
    pool: Weak<Mutex<WorkPool>>,
    id: u64,
    pending: PendingMorselPlanner,
) -> SpawnedTask<()> {
    SpawnedTask::spawn(async move {
        let result = pending.await;
        let Some(pool) = pool.upgrade() else {
            return;
        };
        let wakers = {
            let mut pool = pool.lock();
            pool.completed.push((id, result));
            std::mem::take(&mut pool.wakers)
        };
        for waker in wakers {
            waker.wake();
        }
    })
}
