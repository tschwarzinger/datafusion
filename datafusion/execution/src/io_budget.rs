// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`IoBudget`]: a query-wide bound on scan I/O held in memory.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::task::Waker;

use parking_lot::Mutex;

/// Bounds the bytes a query's scans hold as I/O in flight plus fetched data
/// not yet decoded.
///
/// One budget is shared by all scans (and all their partitions) of a query
/// (see [`TaskContext::io_budget`]), so the bound holds for the query as a
/// whole, and a scan that needs more I/O concurrency can use budget other
/// scans leave idle.
///
/// The limit is a hard cap with one exception that keeps every query making
/// progress: when the budget is exhausted (or smaller than an I/O), a single
/// I/O may still start as long as no other I/O of the query is in flight
/// ([`Self::try_start_sole_io`]). A budget of `0` therefore performs the I/O
/// of a query sequentially, one at a time across all partitions and scans,
/// while fetched data is decoded in parallel.
///
/// Reserving and releasing are lock free (every scan partition of the query
/// does so for each I/O); only waiting for budget takes a lock.
///
/// [`TaskContext::io_budget`]: crate::TaskContext::io_budget
pub struct IoBudget {
    limit: usize,
    reserved: AtomicUsize,
    /// I/Os in flight (see [`IoTicket`]).
    in_flight: AtomicUsize,
    /// Whether `waiters` may be non-empty, so releases skip the lock
    /// otherwise.
    has_waiters: AtomicBool,
    /// Scans waiting for budget.
    waiters: Mutex<Vec<Waker>>,
}

/// Marks one I/O of a query as in flight, from its start until the ticket
/// is dropped (the I/O completed, or was abandoned).
pub struct IoTicket {
    budget: Arc<IoBudget>,
}

impl Drop for IoTicket {
    fn drop(&mut self) {
        if self.budget.in_flight.fetch_sub(1, Ordering::AcqRel) == 1 {
            // a sole I/O may start now
            self.budget.wake_waiters();
        }
    }
}

impl fmt::Debug for IoTicket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IoTicket").finish_non_exhaustive()
    }
}

impl IoBudget {
    /// Create a budget of `limit` bytes.
    pub fn new(limit: usize) -> Arc<Self> {
        Arc::new(Self {
            limit,
            reserved: AtomicUsize::new(0),
            in_flight: AtomicUsize::new(0),
            has_waiters: AtomicBool::new(false),
            waiters: Mutex::default(),
        })
    }

    /// The configured limit, in bytes.
    pub fn limit(&self) -> usize {
        self.limit
    }

    /// Bytes currently reserved.
    pub fn reserved(&self) -> usize {
        self.reserved.load(Ordering::Acquire)
    }

    /// Number of I/Os in flight.
    pub fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::Acquire)
    }

    /// Is any budget left?
    pub fn has_available(&self) -> bool {
        self.reserved() < self.limit
    }

    /// Could an I/O of `bytes` start now, by [`Self::try_start_io`] or
    /// [`Self::try_start_sole_io`]?
    pub fn would_start_io(&self, bytes: usize) -> bool {
        self.reserved() + bytes <= self.limit || self.in_flight() == 0
    }

    /// Start an I/O of `bytes` if they fit within the limit, reserving them.
    pub fn try_start_io(self: &Arc<Self>, bytes: usize) -> Option<IoTicket> {
        self.try_reserve(bytes).then(|| {
            self.in_flight.fetch_add(1, Ordering::AcqRel);
            IoTicket {
                budget: Arc::clone(self),
            }
        })
    }

    /// Start an I/O of `bytes` regardless of the limit if no other I/O of the
    /// query is in flight, reserving them. This keeps a query making progress
    /// when its budget is exhausted, or smaller than a single I/O.
    pub fn try_start_sole_io(self: &Arc<Self>, bytes: usize) -> Option<IoTicket> {
        self.in_flight
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .ok()?;
        self.force_reserve(bytes);
        Some(IoTicket {
            budget: Arc::clone(self),
        })
    }

    /// Reserve `bytes` if they fit within the limit.
    pub fn try_reserve(&self, bytes: usize) -> bool {
        let mut reserved = self.reserved();
        loop {
            if reserved + bytes > self.limit {
                return false;
            }
            match self.reserved.compare_exchange_weak(
                reserved,
                reserved + bytes,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(current) => reserved = current,
            }
        }
    }

    /// Reserve `bytes` regardless of the limit, for data that has already
    /// been fetched.
    pub fn force_reserve(&self, bytes: usize) {
        self.reserved.fetch_add(bytes, Ordering::AcqRel);
    }

    /// Release `bytes` reserved earlier, waking scans waiting for budget.
    pub fn release(&self, bytes: usize) {
        if bytes == 0 {
            return;
        }
        // saturating: releases never exceed reservations, but don't wrap if
        // they ever did
        let mut reserved = self.reserved();
        while let Err(current) = self.reserved.compare_exchange_weak(
            reserved,
            reserved.saturating_sub(bytes),
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            reserved = current;
        }
        self.wake_waiters();
    }

    fn wake_waiters(&self) {
        if self.has_waiters.load(Ordering::Acquire) {
            let waiters = {
                let mut waiters = self.waiters.lock();
                self.has_waiters.store(false, Ordering::Release);
                std::mem::take(&mut *waiters)
            };
            for waker in waiters {
                waker.wake();
            }
        }
    }

    /// Wake `waker` the next time budget is released or the last in-flight
    /// I/O completes.
    pub fn register(&self, waker: &Waker) {
        let mut waiters = self.waiters.lock();
        if !waiters.iter().any(|w| w.will_wake(waker)) {
            waiters.push(waker.clone());
        }
        self.has_waiters.store(true, Ordering::Release);
    }
}

impl fmt::Debug for IoBudget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IoBudget")
            .field("limit", &self.limit)
            .field("reserved", &self.reserved())
            .field("in_flight", &self.in_flight())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::task::Wake;

    struct Count(AtomicUsize);
    impl Wake for Count {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn counting_waker() -> (Arc<Count>, Waker) {
        let count = Arc::new(Count(AtomicUsize::new(0)));
        let waker = Waker::from(Arc::clone(&count));
        (count, waker)
    }

    #[test]
    fn reserve_and_release() {
        let budget = IoBudget::new(100);
        assert!(budget.try_reserve(60));
        assert!(!budget.try_reserve(50));
        assert!(budget.has_available());
        budget.force_reserve(50);
        assert_eq!(budget.reserved(), 110);
        assert!(!budget.has_available());
        budget.release(60);
        assert!(budget.try_reserve(40));
        assert_eq!(budget.reserved(), 90);
    }

    #[test]
    fn io_beyond_the_limit_only_when_none_in_flight() {
        let budget = IoBudget::new(100);
        let a = budget.try_start_io(80).expect("fits");
        assert!(budget.try_start_io(80).is_none(), "exceeds the limit");
        assert!(
            budget.try_start_sole_io(80).is_none(),
            "another is in flight"
        );
        assert!(!budget.would_start_io(80));
        drop(a);
        // the bytes are still reserved (fetched data), but no I/O is in flight
        assert_eq!(budget.reserved(), 80);
        assert!(budget.would_start_io(1000));
        let b = budget.try_start_sole_io(1000).expect("none in flight");
        assert_eq!(budget.reserved(), 1080);
        assert_eq!(budget.in_flight(), 1);
        assert!(budget.try_start_sole_io(1).is_none());
        drop(b);
        assert_eq!(budget.in_flight(), 0);
    }

    #[test]
    fn zero_budget_is_sequential() {
        let budget = IoBudget::new(0);
        // I/O of unknown size reserves nothing and fits
        let unknown = budget.try_start_io(0).expect("fits");
        assert!(budget.try_start_sole_io(10).is_none());
        drop(unknown);
        let first = budget.try_start_sole_io(10).expect("none in flight");
        assert!(budget.try_start_io(10).is_none());
        assert!(budget.try_start_sole_io(10).is_none());
        drop(first);
        assert!(budget.try_start_sole_io(10).is_some());
    }

    #[test]
    fn waiters_are_woken_by_release_and_by_the_last_io() {
        let (count, waker) = counting_waker();
        let budget = IoBudget::new(10);
        let a = budget.try_start_io(10).expect("fits");
        let b = budget.try_start_sole_io(0);
        assert!(b.is_none());
        budget.register(&waker);
        budget.register(&waker); // deduplicated
        budget.release(5);
        assert_eq!(count.0.load(Ordering::Relaxed), 1);
        // waiters are woken once per registration
        budget.release(5);
        assert_eq!(count.0.load(Ordering::Relaxed), 1);
        budget.register(&waker);
        drop(a);
        assert_eq!(count.0.load(Ordering::Relaxed), 2);
        // releases never wrap
        budget.release(100);
        assert_eq!(budget.reserved(), 0);
    }
}
