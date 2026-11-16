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
use super::policy::{Admitted, Policy, PoolLoad};

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
    /// Query-wide budget for in-flight I/O plus fetched but not yet read
    /// morsels (see `policy`). `None`: no prefetch, one unit of work per
    /// stream.
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
///   the pool and wakes a waiting stream, so it completes without any
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
    units: BTreeMap<UnitId, Unit>,
    next_unit: u64,
    next_io: u64,
    /// CPU-ready planners.
    ready_planners: VecDeque<(UnitId, Box<dyn MorselPlanner>)>,
    /// Planners (or files) a stream is currently planning, outside the lock.
    planning: usize,
    /// Planner I/O that has not been started (polled) yet.
    unstarted_io: VecDeque<(UnitId, PendingMorselPlanner)>,
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
    /// Streams waiting for work (by [`Self::new_waiter`] id), in the order
    /// they started waiting; at most one entry per stream.
    wakers: VecDeque<(usize, Waker)>,
    /// Next id handed out by [`Self::new_waiter`].
    next_waiter: usize,
}

/// Identifies a unit of work and defines the planned (read) order of units:
/// files in the order they were opened, and the parts of a split file in the
/// order its planner returned them, right after the file's first part and
/// before any file opened later.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct UnitId {
    file: u64,
    part: u32,
}

/// Book keeping for one unit of work.
struct Unit {
    /// Number of planners, I/Os and morsels (ready or being read) alive for
    /// this unit.
    outstanding: usize,
    /// Part number for the next part split off this unit.
    next_part: u32,
    /// Whether finishing this unit counts as a processed file.
    counts_file: bool,
    /// Number of this unit's morsels being read.
    reading: usize,
}

/// A morsel ready to be read.
pub(super) struct ReadyMorsel {
    pub(super) unit: UnitId,
    pub(super) morsel: Box<dyn Morsel>,
    /// Fetched bytes it holds, if known (`WorkInfo::bytes`).
    pub(super) bytes: Option<usize>,
}

struct InFlightIo {
    id: u64,
    unit: UnitId,
    bytes: Option<usize>,
    future: IoFuture,
}

enum IoFuture {
    /// Polled inline by whichever stream drives the pool (no Tokio runtime
    /// available). Holds its admission while in flight.
    Inline(PendingMorselPlanner, #[expect(dead_code)] Admitted),
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
            wakers: VecDeque::new(),
            next_waiter: 0,
        }
    }

    pub(super) fn new_shared(config: PoolConfig) -> SharedPool {
        Arc::new(Mutex::new(Self::new(config)))
    }

    fn budget(&self) -> Option<&Arc<IoBudget>> {
        self.config.io_budget.as_ref()
    }

    fn release_budget(&self, bytes: usize) {
        if let Some(budget) = self.budget() {
            budget.release(bytes);
        }
    }

    /// The scheduling policy, with this query's budget.
    fn policy(&self) -> Policy<'_> {
        Policy::new(self.budget())
    }

    // ---------------------------------------------------------------
    // Units
    // ---------------------------------------------------------------

    /// A new unit for a file, ordered after all units opened so far.
    fn new_unit(&mut self, counts_file: bool) -> UnitId {
        let id = UnitId {
            file: self.next_unit,
            part: 0,
        };
        self.next_unit += 1;
        self.insert_unit(id, counts_file);
        id
    }

    /// A new unit for a part split off `parent`, ordered after `parent` and
    /// the parts split off it before, but before any later file.
    fn new_part(&mut self, parent: UnitId) -> UnitId {
        let unit = self.units.get_mut(&parent).expect("parent unit is open");
        let id = UnitId {
            file: parent.file,
            part: unit.next_part,
        };
        unit.next_part += 1;
        self.insert_unit(id, false);
        id
    }

    fn insert_unit(&mut self, id: UnitId, counts_file: bool) {
        self.units.insert(
            id,
            Unit {
                outstanding: 0,
                next_part: id.part + 1,
                counts_file,
                reading: 0,
            },
        );
    }

    fn add_outstanding(&mut self, unit: UnitId, n: usize) {
        if let Some(u) = self.units.get_mut(&unit) {
            u.outstanding += n;
        }
    }

    /// Release one outstanding item of `unit`, finishing it at zero.
    pub(super) fn release(&mut self, unit: UnitId, metrics: &mut FileStreamMetrics) {
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
            self.notify();
        }
    }

    /// Number of open units that count as files (for limit accounting).
    pub(super) fn open_files(&self) -> usize {
        self.units.values().filter(|u| u.counts_file).count()
    }

    // ---------------------------------------------------------------
    // Waking
    // ---------------------------------------------------------------

    /// Wake waiting streams after the pool changed.
    ///
    /// While work is pending, only one stream is woken: waking every idle
    /// stream for each completed I/O or read morsel makes them all contend
    /// for this lock, and all but one find nothing to do. The woken stream
    /// wakes the next one if more work is ready (see [`Self::pass_on`]).
    /// Once nothing is pending, all are woken so they can finish.
    fn notify(&mut self) {
        if self.has_pending_work() {
            self.wake_one();
        } else {
            for (_, waker) in self.wakers.drain(..) {
                waker.wake();
            }
        }
    }

    fn wake_one(&mut self) {
        if let Some((_, waker)) = self.wakers.pop_front() {
            waker.wake();
        }
    }

    /// Stream `waiter` no longer waits (it is dropped). If it was woken but
    /// not polled yet, pass the wake-up on.
    pub(super) fn unregister(&mut self, waiter: usize) {
        self.wakers.retain(|(id, _)| *id != waiter);
        self.notify();
    }

    /// An id identifying a stream driving this pool while it waits.
    pub(super) fn new_waiter(&mut self) -> usize {
        self.next_waiter += 1;
        self.next_waiter
    }

    /// Called by stream `waiter` when it drives the pool: it is no longer
    /// waiting (so a wake-up is not wasted on it), and if other streams are
    /// waiting and there is ready work (possibly more than this stream
    /// takes), one of them is woken. This passes a wake-up along the waiting
    /// streams for as long as work is ready, without waking them all at once.
    pub(super) fn pass_on(&mut self, waiter: usize) {
        if self.wakers.is_empty() {
            return;
        }
        self.wakers.retain(|(id, _)| *id != waiter);
        if self.has_ready_work(false) {
            self.wake_one();
        }
    }

    /// Register a stream that found nothing to do.
    ///
    /// With `may_open_more`, the stream could open more work once budget is
    /// available, so it also waits for budget released by other scans of the
    /// query.
    pub(super) fn register(&mut self, waiter: usize, waker: &Waker, may_open_more: bool) {
        match self.wakers.iter_mut().find(|(id, _)| *id == waiter) {
            // the stream may be polled with a different waker than before
            Some((_, registered)) => registered.clone_from(waker),
            None => self.wakers.push_back((waiter, waker.clone())),
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
            if let IoFuture::Inline(future, _) = &mut self.in_flight[i].future
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
                    self.release_budget(b);
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

    /// Index of the I/O in `unstarted_io` to start next: in the order the
    /// I/O was created, or for ordered reads in planned order, so the unit
    /// read next is not fetched after later ones (e.g. when the budget only
    /// lets I/O run one at a time).
    fn next_io_index(&self) -> Option<usize> {
        match self.config.read_order {
            ReadOrder::Any => (!self.unstarted_io.is_empty()).then_some(0),
            ReadOrder::Preferred | ReadOrder::Required => self
                .unstarted_io
                .iter()
                .enumerate()
                .min_by_key(|(_, (unit, _))| *unit)
                .map(|(idx, _)| idx),
        }
    }

    /// Admit the I/O to start next (see [`Self::next_io_index`]) now (see
    /// [`Policy::admit_io`])? Returns its index and admission.
    fn admit_io(&self) -> Option<(usize, Admitted)> {
        let idx = self.next_io_index()?;
        let (_, pending) = &self.unstarted_io[idx];
        let admitted = self.policy().admit_io(self, pending.info().bytes)?;
        Some((idx, admitted))
    }

    /// Would [`Self::admit_io`] admit an I/O now?
    fn can_start_io(&self) -> bool {
        self.next_io_index().is_some_and(|idx| {
            let (_, pending) = &self.unstarted_io[idx];
            self.policy().would_admit_io(self, pending.info().bytes)
        })
    }

    /// Start I/O as allowed by the policy. Returns true if any I/O was
    /// started.
    pub(super) fn start_io(&mut self, this: &SharedPool) -> bool {
        let mut started = false;
        while let Some((idx, admitted)) = self.admit_io() {
            let (unit, pending) = self.unstarted_io.remove(idx).expect("index valid");
            let bytes = pending.info().bytes;
            match bytes {
                Some(b) => self.in_flight_bytes += b,
                None => self.in_flight_unknown += 1,
            }
            let id = self.next_io;
            self.next_io += 1;
            let future = if tokio::runtime::Handle::try_current().is_ok() {
                IoFuture::Spawned(spawn_delivering(
                    Arc::downgrade(this),
                    id,
                    pending,
                    admitted,
                ))
            } else {
                IoFuture::Inline(pending, admitted)
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
        self.plannable_index(stream_reading).is_some()
    }

    /// Index of the ready planner to plan next, if planning is allowed now.
    fn plannable_index(&self, stream_reading: bool) -> Option<usize> {
        if self.ready_planners.is_empty() {
            return None;
        }
        // planners that need not be deferred: in the order they became ready,
        // or for ordered reads in planned order (so the unit read next is
        // planned, and its I/O issued, first)
        let mut undeferred = self
            .ready_planners
            .iter()
            .enumerate()
            .filter(|(_, (_, planner))| !planner.info().defer);
        let next = match self.config.read_order {
            ReadOrder::Any => undeferred.next(),
            ReadOrder::Preferred | ReadOrder::Required => {
                undeferred.min_by_key(|(_, (unit, _))| *unit)
            }
        };
        if let Some((idx, _)) = next {
            return Some(idx);
        }
        // Only deferred planners are ready (see
        // `MorselPlanner::defer_planning`): plan one only when a stream needs
        // its output, i.e. it has nothing to read. With an ordered read that is
        // only the planner of the unit read next: planning a later unit while
        // the next one's I/O is still in flight would evaluate it too early.
        if stream_reading || self.eligible_morsel_index().is_some() {
            return None;
        }
        match self.config.read_order {
            ReadOrder::Any => self
                .ready_planners
                .iter()
                .enumerate()
                .min_by_key(|(_, (unit, _))| *unit)
                .map(|(idx, _)| idx),
            ReadOrder::Preferred | ReadOrder::Required => {
                let next = self.next_unit_to_read()?;
                self.ready_planners
                    .iter()
                    .position(|(unit, _)| *unit == next)
            }
        }
    }

    /// Take a planner to plan outside the lock, if planning is allowed now.
    ///
    /// The caller must hand the result to [`Self::finish_planning`].
    pub(super) fn take_planner(
        &mut self,
        stream_reading: bool,
    ) -> Option<(UnitId, Box<dyn MorselPlanner>)> {
        let idx = self.plannable_index(stream_reading)?;
        let planner = self.ready_planners.remove(idx).expect("index valid");
        self.planning += 1;
        Some(planner)
    }

    /// Apply the result of planning a planner of `unit` taken with
    /// [`Self::take_planner`].
    pub(super) fn finish_planning(
        &mut self,
        unit: UnitId,
        result: Result<Option<MorselPlan>>,
        metrics: &mut FileStreamMetrics,
        skip_errors: bool,
    ) -> PoolStep {
        self.planning -= 1;
        // others may be waiting for planning to finish
        self.notify();
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

        // A planner that split its work into several independent planners.
        // Unless any order will do, give each part a unit of its own, ordered
        // right after `unit`: parts are then fetched concurrently (within the
        // budget) but read strictly in planned order (see
        // `eligible_morsel_index`), so a sorted scan can prefetch several row
        // groups of a file and still release them in order.
        if planners.len() > 1 && self.config.read_order != ReadOrder::Any {
            for planner in planners.split_off(1) {
                let part = self.new_part(unit);
                self.add_outstanding(part, 1);
                self.ready_planners.push_back((part, planner));
            }
        }

        self.add_outstanding(
            unit,
            morsels.len() + planners.len() + usize::from(pending.is_some()),
        );
        for morsel in morsels {
            let bytes = morsel.info().bytes;
            self.buffered_bytes += bytes.unwrap_or(0);
            // already fetched: charged to the budget until read
            if let Some(budget) = self.budget() {
                budget.force_reserve(bytes.unwrap_or(0));
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

    /// May another unit of work (file) be opened (see
    /// [`Policy::may_open_unit`])?
    pub(super) fn may_open_unit(&self) -> bool {
        self.policy().may_open_unit(self)
    }

    /// Register a newly opened file whose planner is being created outside
    /// the lock. Returns its unit; finish with [`Self::finish_open`].
    pub(super) fn begin_open(&mut self) -> UnitId {
        let unit = self.new_unit(true);
        self.add_outstanding(unit, 1);
        self.planning += 1;
        unit
    }

    /// Apply the result of creating the planner for `unit`.
    pub(super) fn finish_open(
        &mut self,
        unit: UnitId,
        planner: Result<Box<dyn MorselPlanner>>,
        metrics: &mut FileStreamMetrics,
        skip_errors: bool,
    ) -> PoolStep {
        self.planning -= 1;
        self.notify();
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

    /// Number of ready morsels that may be read next (all of them, or for
    /// ordered reads only those of the next unit to read).
    fn eligible_morsel_count(&self) -> usize {
        match self.config.read_order {
            ReadOrder::Any => self.ready_morsels.len(),
            _ => match self.eligible_morsel_index() {
                Some(idx) => {
                    let unit = self.ready_morsels[idx].unit;
                    self.ready_morsels.iter().filter(|m| m.unit == unit).count()
                }
                None => 0,
            },
        }
    }

    /// The unit an ordered read reads next: the earliest unit no stream is
    /// reading.
    fn next_unit_to_read(&self) -> Option<UnitId> {
        self.units
            .iter()
            .find(|(_, u)| u.reading == 0)
            .map(|(id, _)| *id)
    }

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
        let next = self.next_unit_to_read()?;
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
        unit: UnitId,
        bytes: usize,
        metrics: &mut FileStreamMetrics,
    ) {
        self.buffered_bytes -= bytes;
        self.release_budget(bytes);
        self.active_readers -= 1;
        if let Some(u) = self.units.get_mut(&unit) {
            u.reading -= 1;
        }
        self.release(unit, metrics);
        // budget was freed, and the next unit may now be eligible
        self.notify();
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
            || self.can_start_io()
            || (!stream_reading && self.eligible_morsel_index().is_some())
    }
}

/// What the pool holds, for the scheduling [`Policy`].
impl PoolLoad for WorkPool {
    fn streams(&self) -> usize {
        self.config.streams
    }
    fn units(&self) -> usize {
        self.units.len()
    }
    fn planning(&self) -> usize {
        self.planning
    }
    fn ready_planners(&self) -> usize {
        self.ready_planners.len()
    }
    fn unstarted_io(&self) -> usize {
        self.unstarted_io.len()
    }
    fn in_flight(&self) -> usize {
        self.in_flight.len()
    }
    fn in_flight_unknown(&self) -> usize {
        self.in_flight_unknown
    }
    fn ready_morsels(&self) -> usize {
        self.ready_morsels.len()
    }
    fn eligible_morsels(&self) -> usize {
        self.eligible_morsel_count()
    }
    fn has_unsized_morsel(&self) -> bool {
        self.ready_morsels.iter().any(|m| m.bytes.is_none())
    }
    fn readers(&self) -> usize {
        self.active_readers
    }
}

impl Drop for WorkPool {
    /// Return this scan's reservations to the query-wide budget, e.g. when
    /// the scan is dropped early (LIMIT, cancellation).
    fn drop(&mut self) {
        self.release_budget(self.in_flight_bytes + self.buffered_bytes);
    }
}

/// Spawn `pending` onto the Tokio runtime. On completion its result is
/// delivered into the pool and a waiting stream is woken, so the I/O
/// completes without any particular stream being polled.
fn spawn_delivering(
    pool: Weak<Mutex<WorkPool>>,
    id: u64,
    pending: PendingMorselPlanner,
    admitted: Admitted,
) -> SpawnedTask<()> {
    SpawnedTask::spawn(async move {
        let result = pending.await;
        // the I/O is no longer in flight (its bytes stay reserved)
        drop(admitted);
        let Some(pool) = pool.upgrade() else {
            return;
        };
        let waker = {
            let mut pool = pool.lock();
            pool.completed.push((id, result));
            pool.wakers.pop_front()
        };
        if let Some((_, waker)) = waker {
            waker.wake();
        }
    })
}
