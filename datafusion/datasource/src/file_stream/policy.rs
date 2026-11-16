// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Scheduling policy of the [`WorkPool`]: how much work to start ahead of
//! decoding.
//!
//! The pool does the book keeping (which items exist, their order) and
//! answers the questions of [`PoolLoad`]; [`Policy`] decides from those
//! answers, and the query-wide [`IoBudget`] if any, when another I/O may start
//! and when another file may be opened.
//! Keeping the rules here keeps them in one place and testable without a
//! pool.
//!
//! [`WorkPool`]: super::work_pool::WorkPool

use std::sync::Arc;

use datafusion_execution::{IoBudget, IoTicket};

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

/// Decides how much work a pool starts.
///
/// * Without a budget, each stream has one unit of work: an I/O in flight, a
///   morsel ready to read, or a morsel being read. Each partition alternates
///   between fetching and decoding, as a scan without prefetch does.
/// * With a query-wide [`IoBudget`], I/O starts while it fits the budget
///   (prefetch). When it does not fit, an I/O only starts if no other I/O of
///   the query is in flight and this pool has no fetched morsel that could be
///   read now (see [`IoBudget::try_start_sole_io`]): the query keeps making
///   progress (e.g. when another scan, not being polled, holds the budget),
///   I/O larger than the budget still runs, and fetched data beyond the
///   budget stays bounded. Morsels that may not be read yet (an ordered
///   read waiting for an earlier unit) don't block it: the earlier unit's
///   I/O may be the one needed.
#[derive(Debug, Clone, Copy)]
pub struct Policy<'a> {
    budget: Option<&'a Arc<IoBudget>>,
}

impl<'a> Policy<'a> {
    pub fn new(budget: Option<&'a Arc<IoBudget>>) -> Self {
        Self { budget }
    }

    /// Start an I/O of `bytes` (if known) now? Returns the ticket marking it
    /// in flight if so (reserving its known size from the budget, if any).
    pub fn admit_io(
        &self,
        load: &impl PoolLoad,
        bytes: Option<usize>,
    ) -> Option<Admitted> {
        let bytes = bytes.unwrap_or(0);
        match self.budget {
            None => below_minimum(load).then_some(Admitted(None)),
            Some(budget) => budget
                .try_start_io(bytes)
                .or_else(|| {
                    (load.eligible_morsels() == 0)
                        .then(|| budget.try_start_sole_io(bytes))
                        .flatten()
                })
                .map(|ticket| Admitted(Some(ticket))),
        }
    }

    /// Would [`Self::admit_io`] start the I/O now?
    pub fn would_admit_io(&self, load: &impl PoolLoad, bytes: Option<usize>) -> bool {
        let bytes = bytes.unwrap_or(0);
        match self.budget {
            None => below_minimum(load),
            Some(budget) => {
                budget.reserved() + bytes <= budget.limit()
                    || (load.eligible_morsels() == 0 && budget.in_flight() == 0)
            }
        }
    }

    /// May another unit of work (file) be opened?
    pub fn may_open_unit(&self, load: &impl PoolLoad) -> bool {
        let streams = load.streams().max(1);
        // Nothing open, or fewer items of work than streams (like each
        // partition opening its own file).
        let work_items = load.planning()
            + load.ready_planners()
            + load.unstarted_io()
            + load.in_flight()
            + load.ready_morsels()
            + load.readers();
        if work_items < streams {
            return true;
        }
        let Some(budget) = self.budget else {
            return false;
        };
        // Prefetch beyond that, within the budget.
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
            && budget.has_available()
    }
}

/// An admitted I/O: holds its [`IoTicket`] (if budgeted) while in flight.
#[derive(Debug)]
pub struct Admitted(#[expect(dead_code)] Option<IoTicket>);

/// Without a budget: does the pool have fewer units of work than streams?
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
        eligible_morsels: usize,
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
            self.eligible_morsels
        }
        fn eligible_morsels(&self) -> usize {
            self.eligible_morsels
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
    fn without_budget_one_unit_of_work_per_stream() {
        let policy = Policy::new(None);
        let mut l = load(2);
        assert!(policy.admit_io(&l, Some(100)).is_some());
        l.in_flight = 1;
        assert!(policy.admit_io(&l, Some(100)).is_some());
        l.in_flight = 2;
        assert!(policy.admit_io(&l, Some(100)).is_none());
        // a stream reading counts as its unit of work
        l.in_flight = 1;
        l.readers = 1;
        assert!(policy.admit_io(&l, Some(100)).is_none());
        assert!(!policy.would_admit_io(&l, Some(100)));
    }

    #[test]
    fn within_the_budget_then_one_io_at_a_time() {
        let budget = IoBudget::new(150);
        let policy = Policy::new(Some(&budget));
        let l = load(4);
        let a = policy.admit_io(&l, Some(100)).expect("fits");
        assert!(policy.admit_io(&l, Some(100)).is_none(), "200 > 150");
        let b = policy.admit_io(&l, Some(50)).expect("fits");
        assert!(!policy.would_admit_io(&l, Some(1)));
        drop((a, b));
        // the data is fetched (still reserved), no I/O in flight: one more
        assert!(policy.would_admit_io(&l, Some(100)));
        let c = policy.admit_io(&l, Some(100)).expect("sole I/O");
        assert!(policy.admit_io(&l, Some(1)).is_none());
        drop(c);
        // not while fetched data of this pool can be read
        let mut waiting = load(4);
        waiting.eligible_morsels = 1;
        assert!(!policy.would_admit_io(&waiting, Some(1)));
        assert!(policy.admit_io(&waiting, Some(1)).is_none());
        assert!(policy.admit_io(&l, Some(1)).is_some());
    }

    #[test]
    fn zero_budget_is_sequential() {
        let budget = IoBudget::new(0);
        let policy = Policy::new(Some(&budget));
        let l = load(4);
        let a = policy.admit_io(&l, Some(100)).expect("first I/O");
        assert!(policy.admit_io(&l, Some(100)).is_none());
        drop(a);
        assert!(policy.admit_io(&l, Some(100)).is_some());
    }

    #[test]
    fn opening_files() {
        let budget = IoBudget::new(1000);
        let mut l = load(2);
        assert!(Policy::new(Some(&budget)).may_open_unit(&l));
        // one file per stream regardless of the budget
        l.units = 1;
        l.in_flight = 1;
        assert!(Policy::new(None).may_open_unit(&l));
        // beyond: only with budget, and not while metadata of unknown size
        // is read for each stream
        l.units = 2;
        l.in_flight = 2;
        assert!(!Policy::new(None).may_open_unit(&l));
        assert!(!Policy::new(Some(&IoBudget::new(0))).may_open_unit(&l));
        assert!(Policy::new(Some(&budget)).may_open_unit(&l));
        l.in_flight_unknown = 2;
        assert!(!Policy::new(Some(&budget)).may_open_unit(&l));
        l.in_flight_unknown = 0;
        l.unsized_morsel = true;
        assert!(!Policy::new(Some(&budget)).may_open_unit(&l));
    }
}
