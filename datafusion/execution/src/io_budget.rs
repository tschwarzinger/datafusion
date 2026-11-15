// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`IoBudget`]: a query-wide bound on prefetched I/O.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::task::Waker;

use parking_lot::Mutex;

/// Bounds the bytes a query keeps in flight or fetched-but-not-yet-decoded
/// when its scans prefetch I/O ahead of decoding.
///
/// One budget is shared by all scans (and all their partitions) of a query
/// (see [`TaskContext::io_budget`]), so the bound holds for the query as a
/// whole, and a scan that needs more I/O concurrency can use budget other
/// scans leave idle.
///
/// The budget only limits prefetch *beyond* a guaranteed minimum: each
/// partition of a scan may always have one unit of work (an I/O in flight, or
/// fetched data being decoded), as it does without a budget, even when the
/// budget is exhausted or smaller than a single I/O. So a scan is never slower
/// than with no prefetch at all, makes progress when other scans hold budget
/// but are not being polled (e.g. the probe side of a join while its build
/// side is collected), and I/O larger than the budget still runs.
///
/// [`TaskContext::io_budget`]: crate::TaskContext::io_budget
///
/// Reserving and releasing are lock free (every scan partition of the query
/// does so for each I/O); only waiting for budget takes a lock.
pub struct IoBudget {
    limit: usize,
    reserved: AtomicUsize,
    /// Whether `waiters` may be non-empty, so releases skip the lock
    /// otherwise.
    has_waiters: AtomicBool,
    /// Scans waiting for budget.
    waiters: Mutex<Vec<Waker>>,
}

impl IoBudget {
    /// Create a budget of `limit` bytes. A limit of `0` disables prefetching.
    pub fn new(limit: usize) -> Arc<Self> {
        Arc::new(Self {
            limit,
            reserved: AtomicUsize::new(0),
            has_waiters: AtomicBool::new(false),
            waiters: Mutex::default(),
        })
    }

    /// The configured limit, in bytes.
    pub fn limit(&self) -> usize {
        self.limit
    }

    /// Is prefetching enabled (non-zero limit)?
    pub fn is_enabled(&self) -> bool {
        self.limit > 0
    }

    /// Bytes currently reserved.
    pub fn reserved(&self) -> usize {
        self.reserved.load(Ordering::Acquire)
    }

    /// Is any budget left?
    pub fn has_available(&self) -> bool {
        self.reserved() < self.limit
    }

    /// Would `bytes` fit within the limit now?
    pub fn fits(&self, bytes: usize) -> bool {
        self.reserved() + bytes <= self.limit
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

    /// Reserve `bytes` regardless of the limit, e.g. for I/O that is always
    /// allowed to make progress, or for data that has already been fetched.
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

    /// Wake `waker` the next time budget is released.
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
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn release_wakes_waiters_once() {
        use std::sync::atomic::AtomicUsize;
        use std::task::Wake;

        struct Count(AtomicUsize);
        impl Wake for Count {
            fn wake(self: Arc<Self>) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }
        let count = Arc::new(Count(AtomicUsize::new(0)));
        let waker = Waker::from(Arc::clone(&count));

        let budget = IoBudget::new(10);
        budget.force_reserve(20);
        budget.register(&waker);
        budget.register(&waker); // deduplicated
        budget.release(5);
        assert_eq!(count.0.load(Ordering::Relaxed), 1);
        // waiters are woken once per registration
        budget.release(5);
        assert_eq!(count.0.load(Ordering::Relaxed), 1);
        assert_eq!(budget.reserved(), 10);
        // releases never wrap
        budget.release(100);
        assert_eq!(budget.reserved(), 0);
    }
}
