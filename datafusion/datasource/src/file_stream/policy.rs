// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Scheduling policy of the [`WorkPool`]: how much work to start ahead of
//! decoding.
//!
//! The pool does the book keeping (which items exist, their order) and
//! answers the questions of [`PoolLoad`]; [`Policy`] decides from those
//! answers when another I/O may start and when another file may be opened.
//! Keeping the rules here keeps them in one place and testable without a
//! pool.
//!
//! [`WorkPool`]: super::work_pool::WorkPool

use datafusion_execution::IoBudget;

/// Maximum number of units of work open per stream sharing a pool,
/// regardless of the I/O budget.
const MAX_OPEN_UNITS_PER_STREAM: usize = 64;

/// What a pool currently holds, as far as the [`Policy`] needs to know.
pub trait PoolLoad {
    /// Streams driving the pool.
    fn streams(&self) -> usize;
    /// Open units of work (files or parts of files).
    fn units(&self) -> usize;
    /// Planners and files being planned (CPU, outside the pool's lock).
    fn planning(&self) -> usize;
    /// CPU-ready planners.
    fn ready_planners(&self) -> usize;
    /// I/O not started yet.
    fn unstarted_io(&self) -> usize;
    /// Started I/O.
    fn in_flight(&self) -> usize;
    /// Started I/O of unknown size (e.g. metadata reads).
    fn in_flight_unknown(&self) -> usize;
    /// Fetched morsels ready to be read.
    fn ready_morsels(&self) -> usize;
    /// Of those, the ones that may be read next (all, or for ordered reads
    /// those of the unit read next).
    fn eligible_morsels(&self) -> usize;
    /// Does a ready morsel not report its size?
    fn has_unsized_morsel(&self) -> bool;
    /// Morsels being read.
    fn readers(&self) -> usize;
}

/// Decides how much work a pool starts, within a query-wide [`IoBudget`].
///
/// Every scan gets a guaranteed minimum of one unit of work per stream (an
/// I/O in flight, a morsel ready to read, or a morsel being read): what each
/// partition holds without prefetch, alternating between fetching and
/// decoding. Within the minimum, work starts regardless of the budget: a
/// budget, however small, is never slower than none, a scan makes progress
/// when other scans of the query hold the whole budget, and an I/O larger
/// than the budget still runs. Beyond the minimum, work starts if it fits the
/// budget (prefetch).
#[derive(Debug, Clone, Copy)]
pub struct Policy<'a> {
    budget: &'a IoBudget,
}

impl<'a> Policy<'a> {
    pub fn new(budget: &'a IoBudget) -> Self {
        Self { budget }
    }

    /// May an I/O of `bytes` (if known) start now? If so, its known size is
    /// reserved from the budget.
    pub fn admit_io(&self, load: &impl PoolLoad, bytes: Option<usize>) -> bool {
        let bytes = bytes.unwrap_or(0);
        if below_minimum(load) {
            self.budget.force_reserve(bytes);
            return true;
        }
        self.budget.try_reserve(bytes)
    }

    /// Like [`Self::admit_io`], without reserving.
    pub fn would_admit_io(&self, load: &impl PoolLoad, bytes: Option<usize>) -> bool {
        below_minimum(load) || self.budget.fits(bytes.unwrap_or(0))
    }

    /// May another unit of work (file) be opened?
    pub fn may_open_unit(&self, load: &impl PoolLoad) -> bool {
        let streams = load.streams().max(1);
        // Nothing open, or the guaranteed minimum (one item of work per
        // stream, like each partition opening its own file without a budget).
        let work_items = load.planning()
            + load.ready_planners()
            + load.unstarted_io()
            + load.in_flight()
            + load.ready_morsels()
            + load.readers();
        if work_items < streams {
            return true;
        }
        // Prefetch beyond the minimum, within the budget.
        load.units() < MAX_OPEN_UNITS_PER_STREAM * streams
            // the budget is not already saturated by work waiting to start
            && load.unstarted_io() == 0
            && load.ready_planners() == 0
            // don't open files in bursts while their I/O size is unknown: at
            // most one such file (being planned or loading metadata) per
            // stream
            && load.in_flight_unknown() + load.planning() < streams
            // a morsel that does not report its size (e.g. the `FileOpener`
            // adapter) can't be charged against the budget: don't open more
            // work while such a morsel is waiting
            && !load.has_unsized_morsel()
            && self.budget.has_available()
    }
}

/// Is the pool below the guaranteed minimum (see [`Policy`])?
fn below_minimum(load: &impl PoolLoad) -> bool {
    load.in_flight() + load.eligible_morsels() + load.readers() < load.streams().max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Load {
        streams: usize,
        units: usize,
        in_flight: usize,
        in_flight_unknown: usize,
        readers: usize,
        unsized_morsel: bool,
    }

    impl PoolLoad for Load {
        fn streams(&self) -> usize {
            self.streams
        }
        fn units(&self) -> usize {
            self.units
        }
        fn planning(&self) -> usize {
            0
        }
        fn ready_planners(&self) -> usize {
            0
        }
        fn unstarted_io(&self) -> usize {
            0
        }
        fn in_flight(&self) -> usize {
            self.in_flight
        }
        fn in_flight_unknown(&self) -> usize {
            self.in_flight_unknown
        }
        fn ready_morsels(&self) -> usize {
            0
        }
        fn eligible_morsels(&self) -> usize {
            0
        }
        fn has_unsized_morsel(&self) -> bool {
            self.unsized_morsel
        }
        fn readers(&self) -> usize {
            self.readers
        }
    }

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
        let policy = Policy::new(&budget);
        let mut l = load(2);
        assert!(policy.admit_io(&l, Some(100)));
        l.in_flight = 1;
        assert!(policy.admit_io(&l, Some(100)));
        l.in_flight = 2;
        assert!(!policy.admit_io(&l, Some(100)));
        // a stream reading counts as its unit of work
        l.in_flight = 1;
        l.readers = 1;
        assert!(!policy.admit_io(&l, Some(100)));
        assert_eq!(budget.reserved(), 200, "the minimum is charged too");
    }

    #[test]
    fn prefetch_beyond_the_minimum_within_the_budget() {
        let budget = IoBudget::new(150);
        let policy = Policy::new(&budget);
        let mut l = load(1);
        l.readers = 1;
        assert!(policy.admit_io(&l, Some(100)));
        l.in_flight = 1;
        assert!(!policy.admit_io(&l, Some(100)), "200 > 150");
        assert!(policy.admit_io(&l, Some(50)));
        assert!(!policy.would_admit_io(&l, Some(1)));
    }

    #[test]
    fn opening_files() {
        let budget = IoBudget::new(1000);
        let no_budget = IoBudget::new(0);
        let mut l = load(2);
        assert!(Policy::new(&budget).may_open_unit(&l));
        // one file per stream regardless of the budget
        l.units = 1;
        l.in_flight = 1;
        assert!(Policy::new(&no_budget).may_open_unit(&l));
        // beyond: only with budget, and not while metadata of unknown size
        // is read for each stream
        l.units = 2;
        l.in_flight = 2;
        assert!(!Policy::new(&no_budget).may_open_unit(&l));
        assert!(Policy::new(&budget).may_open_unit(&l));
        l.in_flight_unknown = 2;
        assert!(!Policy::new(&budget).may_open_unit(&l));
        l.in_flight_unknown = 0;
        l.unsized_morsel = true;
        assert!(!Policy::new(&budget).may_open_unit(&l));
    }
}
