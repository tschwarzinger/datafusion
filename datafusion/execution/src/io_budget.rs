// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`IoBudget`]: a query-wide bound on prefetched I/O.

use std::fmt;
use std::sync::Arc;
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
/// The budget only gates *starting* additional I/O: a scan with nothing in
/// flight and nothing being decoded may always start one I/O, even when the
/// budget is exhausted by other scans. This guarantees progress (and that no
/// scan is slower than with no prefetch at all) when other scans hold budget
/// but are not being polled, e.g. the probe side of a join while its build
/// side is collected.
///
/// [`TaskContext::io_budget`]: crate::TaskContext::io_budget
pub struct IoBudget {
    limit: usize,
    state: Mutex<IoBudgetState>,
}

#[derive(Default)]
struct IoBudgetState {
    reserved: usize,
    /// Scans waiting for budget.
    waiters: Vec<Waker>,
}

impl IoBudget {
    /// Create a budget of `limit` bytes. A limit of `0` disables prefetching.
    pub fn new(limit: usize) -> Arc<Self> {
        Arc::new(Self {
            limit,
            state: Mutex::default(),
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
        self.state.lock().reserved
    }

    /// Is any budget left?
    pub fn has_available(&self) -> bool {
        self.state.lock().reserved < self.limit
    }

    /// Reserve `bytes` if they fit within the limit.
    pub fn try_reserve(&self, bytes: usize) -> bool {
        let mut state = self.state.lock();
        if state.reserved + bytes <= self.limit {
            state.reserved += bytes;
            true
        } else {
            false
        }
    }

    /// Reserve `bytes` regardless of the limit, e.g. for I/O that is always
    /// allowed to make progress, or for data that has already been fetched.
    pub fn force_reserve(&self, bytes: usize) {
        self.state.lock().reserved += bytes;
    }

    /// Release `bytes` reserved earlier, waking scans waiting for budget.
    pub fn release(&self, bytes: usize) {
        if bytes == 0 {
            return;
        }
        let waiters = {
            let mut state = self.state.lock();
            state.reserved = state.reserved.saturating_sub(bytes);
            std::mem::take(&mut state.waiters)
        };
        for waker in waiters {
            waker.wake();
        }
    }

    /// Wake `waker` the next time budget is released.
    pub fn register(&self, waker: &Waker) {
        let mut state = self.state.lock();
        if !state.waiters.iter().any(|w| w.will_wake(waker)) {
            state.waiters.push(waker.clone());
        }
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
}
