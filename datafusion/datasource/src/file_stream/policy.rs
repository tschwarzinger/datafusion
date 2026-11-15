// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Scheduling policy of the [`WorkPool`]: pure decisions about how much work
//! to start ahead of decoding, made from a snapshot of what the pool holds.
//!
//! The pool does the book keeping (which items exist, their order); this
//! module decides when another I/O may start and when another file may be
//! opened. Keeping the rules here, as functions of a [`Load`], keeps them in
//! one place and testable without a pool.
//!
//! [`WorkPool`]: super::work_pool::WorkPool

use datafusion_execution::IoBudget;

/// Maximum number of units of work open per stream sharing a pool,
/// regardless of the I/O budget.
const MAX_OPEN_UNITS_PER_STREAM: usize = 64;

/// What a pool currently holds, as seen by the scheduling policy.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct Load {
    /// Streams driving the pool.
    pub(super) streams: usize,
    /// Open units of work (files or parts of files).
    pub(super) units: usize,
    /// Planners and files being planned (CPU, outside the pool's lock).
    pub(super) planning: usize,
    /// CPU-ready planners.
    pub(super) ready_planners: usize,
    /// I/O not started yet.
    pub(super) unstarted_io: usize,
    /// Started I/O.
    pub(super) in_flight: usize,
    /// Started I/O of unknown size (e.g. metadata reads).
    pub(super) in_flight_unknown: usize,
    /// Fetched morsels ready to be read.
    pub(super) ready_morsels: usize,
    /// Of those, the ones that may be read next (all, or for ordered reads
    /// those of the unit read next).
    pub(super) eligible_morsels: usize,
    /// Whether a ready morsel does not report its size.
    pub(super) unsized_morsel: bool,
    /// Morsels being read.
    pub(super) readers: usize,
}

impl Load {
    /// Units of work per stream that every scan has, budget or not: an I/O
    /// in flight, a morsel ready to read, or a morsel being read. This is
    /// what each partition holds without prefetch (it alternates between
    /// fetching and decoding).
    fn below_minimum(&self) -> bool {
        self.in_flight + self.eligible_morsels + self.readers < self.streams.max(1)
    }
}

/// May an I/O of `bytes` (if known) start now? If so, its known size is
/// reserved from `budget`.
///
/// Within the guaranteed minimum (see [`Load`]) the I/O always starts,
/// regardless of the budget: a budget, however small, is never slower than
/// none, a scan makes progress when other scans of the query hold the whole
/// budget, and an I/O larger than the budget still runs. Beyond the minimum,
/// it starts if it fits the query's budget (prefetch).
pub(super) fn admit_io(load: &Load, bytes: Option<usize>, budget: &IoBudget) -> bool {
    let bytes = bytes.unwrap_or(0);
    if load.below_minimum() {
        budget.force_reserve(bytes);
        return true;
    }
    budget.try_reserve(bytes)
}

/// Like [`admit_io`], without reserving.
pub(super) fn would_admit_io(
    load: &Load,
    bytes: Option<usize>,
    budget: &IoBudget,
) -> bool {
    load.below_minimum() || budget.fits(bytes.unwrap_or(0))
}

/// May another unit of work (file) be opened?
pub(super) fn may_open_unit(load: &Load, budget: &IoBudget) -> bool {
    // Nothing open, or the guaranteed minimum (one item of work per stream,
    // like each partition opening its own file without a budget).
    let work_items = load.planning
        + load.ready_planners
        + load.unstarted_io
        + load.in_flight
        + load.ready_morsels
        + load.readers;
    if work_items < load.streams.max(1) {
        return true;
    }
    // Prefetch beyond the minimum, within the budget.
    load.units < MAX_OPEN_UNITS_PER_STREAM * load.streams.max(1)
        // the budget is not already saturated by work waiting to start
        && load.unstarted_io == 0
        && load.ready_planners == 0
        // don't open files in bursts while their I/O size is unknown: at most
        // one such file (being planned or loading metadata) per stream
        && load.in_flight_unknown + load.planning < load.streams.max(1)
        // a morsel that does not report its size (e.g. the `FileOpener`
        // adapter) can't be charged against the budget: don't open more work
        // while such a morsel is waiting
        && !load.unsized_morsel
        && budget.has_available()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(streams: usize) -> Load {
        Load {
            streams,
            ..Default::default()
        }
    }

    #[test]
    fn minimum_is_one_unit_of_work_per_stream() {
        // no budget at all: one I/O per stream, then nothing
        let budget = IoBudget::new(0);
        let mut l = load(2);
        assert!(admit_io(&l, Some(100), &budget));
        l.in_flight = 1;
        assert!(admit_io(&l, Some(100), &budget));
        l.in_flight = 2;
        assert!(!admit_io(&l, Some(100), &budget));
        // a stream reading counts as its unit of work
        l.in_flight = 1;
        l.readers = 1;
        assert!(!admit_io(&l, Some(100), &budget));
        assert_eq!(budget.reserved(), 200, "the minimum is charged too");
    }

    #[test]
    fn prefetch_beyond_the_minimum_within_the_budget() {
        let budget = IoBudget::new(150);
        let mut l = load(1);
        l.readers = 1;
        assert!(admit_io(&l, Some(100), &budget));
        l.in_flight = 1;
        assert!(!admit_io(&l, Some(100), &budget), "200 > 150");
        assert!(admit_io(&l, Some(50), &budget));
        assert!(!would_admit_io(&l, Some(1), &budget));
    }

    #[test]
    fn opening_files() {
        let budget = IoBudget::new(1000);
        let mut l = load(2);
        assert!(may_open_unit(&l, &budget));
        // one file per stream regardless of the budget
        l.units = 1;
        l.in_flight = 1;
        assert!(may_open_unit(&l, &IoBudget::new(0)));
        // beyond: only with budget, and not while metadata of unknown size
        // is read for each stream
        l.units = 2;
        l.in_flight = 2;
        assert!(!may_open_unit(&l, &IoBudget::new(0)));
        assert!(may_open_unit(&l, &budget));
        l.in_flight_unknown = 2;
        assert!(!may_open_unit(&l, &budget));
        l.in_flight_unknown = 0;
        l.unsized_morsel = true;
        assert!(!may_open_unit(&l, &budget));
    }
}
