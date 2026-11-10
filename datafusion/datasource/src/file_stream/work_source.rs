// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::VecDeque;
use std::sync::Arc;
use std::task::Waker;

use crate::PartitionedFile;
use crate::file_groups::FileGroup;
use crate::file_scan_config::FileScanConfig;
use crate::morsel::{Morsel, MorselPlanner};
use parking_lot::Mutex;

/// A unit of work handed out by a [`WorkSource`].
pub(super) enum WorkItem {
    /// An unopened file that still needs to be planned.
    File(Box<PartitionedFile>),
    /// A CPU-ready planner for part of a file that a sibling stream already
    /// opened and split (e.g. a single Parquet row group), donated so idle
    /// siblings can steal it without re-planning the file.
    Planner(Box<dyn MorselPlanner>),
}

/// Result of [`WorkSource::poll_next`].
pub(super) enum NextWork {
    /// A new item of work.
    Item(WorkItem),
    /// No work is queued right now, but a sibling is still planning a file and
    /// may donate more work. The caller's waker has been registered.
    Wait,
    /// No more work.
    Done,
}

/// Source of work for `ScanState`.
///
/// Streams that may share work across siblings use [`WorkSource::Shared`],
/// while streams that can not share work (e.g. because they must preserve file
/// order) use  [`WorkSource::Local`].
#[derive(Debug, Clone)]
pub(super) enum WorkSource {
    /// Files this stream will plan locally without sharing them.
    Local(VecDeque<PartitionedFile>),
    /// Files shared with sibling streams.
    Shared(SharedWorkSource),
}

impl WorkSource {
    /// Pop the next item of work from this work source.
    ///
    /// `waker` is registered when a sibling may still donate work (see
    /// [`NextWork::Wait`]).
    /// When `track_planning` is set, a file taken from a shared queue counts
    /// as "planning" until the caller calls [`SharedWorkSource::settle`].
    pub(super) fn poll_next(&mut self, waker: &Waker, track_planning: bool) -> NextWork {
        match self {
            Self::Local(files) => match files.pop_front() {
                Some(file) => NextWork::Item(WorkItem::File(Box::new(file))),
                None => NextWork::Done,
            },
            Self::Shared(shared) => shared.poll_next(waker, track_planning),
        }
    }

    /// Return the shared work source, if any.
    pub(super) fn shared(&self) -> Option<&SharedWorkSource> {
        match self {
            Self::Local(_) => None,
            Self::Shared(shared) => Some(shared),
        }
    }

    /// Return how many queued files should be counted as already processed
    /// when this stream stops early after hitting a global limit.
    pub(super) fn skipped_on_limit(&self) -> usize {
        match self {
            Self::Local(files) => files.len(),
            Self::Shared(_) => 0,
        }
    }
}

/// Shared source of work for sibling `FileStream`s
///
/// The queue is created once per execution and shared by all reorderable
/// sibling streams for that execution. Whichever stream becomes idle first may
/// take the next unit of work from the front of the queue.
///
/// Besides unopened files, streams may donate CPU-ready planners for parts of
/// a file they already opened (see [`WorkItem::Planner`]). Donated planners
/// are pushed to the front of the queue so that files that are already open
/// are finished before new files are started.
///
/// It uses a [`Mutex`] internally to provide thread-safe access
/// to the shared file queue.
#[derive(Debug, Clone)]
pub(crate) struct SharedWorkSource {
    inner: Arc<SharedWorkSourceInner>,
}

#[derive(Default)]
pub(super) struct SharedWorkSourceInner {
    state: Mutex<SharedState>,
}

impl std::fmt::Debug for SharedWorkSourceInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.lock();
        f.debug_struct("SharedWorkSourceInner")
            .field("files", &state.files)
            .field("planners", &state.planners.len())
            .field("morsels", &state.morsels.len())
            .field("planning", &state.planning)
            .finish()
    }
}

/// A fetched morsel published to the shared pool by the stream that fetched
/// it (its owner), so that any sibling stream with an idle decoder can read
/// it.
pub(super) struct PooledMorsel {
    pub(super) morsel: Box<dyn Morsel>,
    /// `Morsel::buffered_bytes`, charged to the owner's I/O budget until the
    /// morsel is taken.
    pub(super) bytes: usize,
    /// The owner's unit of work this morsel belongs to.
    pub(super) unit: u64,
    /// Where to notify the owner once the morsel is taken.
    pub(super) owner: Arc<MorselInbox>,
}

/// Notifications for a stream that some of its pooled morsels were taken,
/// which releases their bytes from its I/O budget and may finish its units.
#[derive(Default)]
pub(super) struct MorselInbox {
    taken: Mutex<Vec<(u64, usize)>>,
    waker: Mutex<Option<Waker>>,
}

impl MorselInbox {
    /// Record that the morsel of `unit` holding `bytes` was taken, and wake
    /// the owner so it can use the freed budget.
    pub(super) fn notify(&self, unit: u64, bytes: usize) {
        self.taken.lock().push((unit, bytes));
        if let Some(waker) = self.waker.lock().take() {
            waker.wake();
        }
    }

    /// Register the owner's waker.
    pub(super) fn register(&self, waker: &Waker) {
        let mut slot = self.waker.lock();
        if !slot.as_ref().is_some_and(|w| w.will_wake(waker)) {
            *slot = Some(waker.clone());
        }
    }

    /// Take all notifications.
    pub(super) fn drain(&self) -> Vec<(u64, usize)> {
        std::mem::take(&mut *self.taken.lock())
    }
}

#[derive(Default)]
struct SharedState {
    /// Fetched morsels, ready to be read by any stream.
    morsels: VecDeque<PooledMorsel>,
    /// Donated planners, handed out before any unopened file.
    planners: VecDeque<Box<dyn MorselPlanner>>,
    /// Unopened files.
    files: VecDeque<PartitionedFile>,
    /// Number of files taken from the queue whose planners may still donate
    /// work (see [`SharedWorkSource::settle`]).
    planning: usize,
    /// Streams waiting for donated work.
    wakers: Vec<Waker>,
}

impl SharedState {
    fn wake_all(&mut self) {
        for waker in self.wakers.drain(..) {
            waker.wake();
        }
    }
}

impl SharedWorkSource {
    /// Create a shared work source containing the provided unopened files.
    pub(crate) fn new(files: impl IntoIterator<Item = PartitionedFile>) -> Self {
        let files = files.into_iter().collect();
        Self {
            inner: Arc::new(SharedWorkSourceInner {
                state: Mutex::new(SharedState {
                    files,
                    ..Default::default()
                }),
            }),
        }
    }

    /// Create a shared work source for the unopened files in `config`.
    ///
    /// Files are reordered by the file source (e.g. by statistics for TopK)
    /// before being placed in the shared queue, so the most promising files
    /// are processed first across all partitions.
    pub(crate) fn from_config(config: &FileScanConfig) -> Self {
        let files: Vec<_> = config
            .file_groups
            .iter()
            .flat_map(FileGroup::iter)
            .cloned()
            .collect();
        let files = config.file_source.reorder_files(files);
        Self::new(files)
    }

    /// Pop the next unit of work from the shared work queue.
    ///
    /// Taking a file marks it as "planning" until [`Self::settle`] is called
    /// for it: while any file is planning, idle siblings wait (instead of
    /// finishing) because the planner may still donate work.
    fn poll_next(&self, waker: &Waker, track_planning: bool) -> NextWork {
        let mut state = self.inner.state.lock();
        if let Some(planner) = state.planners.pop_front() {
            return NextWork::Item(WorkItem::Planner(planner));
        }
        if let Some(file) = state.files.pop_front() {
            if track_planning {
                state.planning += 1;
            }
            return NextWork::Item(WorkItem::File(Box::new(file)));
        }
        if state.planning > 0 {
            if !state.wakers.iter().any(|w| w.will_wake(waker)) {
                state.wakers.push(waker.clone());
            }
            return NextWork::Wait;
        }
        NextWork::Done
    }

    /// Donate planners for parts of an already opened file to sibling streams.
    pub(super) fn donate(
        &self,
        planners: impl IntoIterator<Item = Box<dyn MorselPlanner>>,
    ) {
        let mut state = self.inner.state.lock();
        // push to the front, keeping the donated planners in order
        let mut planners: Vec<_> = planners.into_iter().collect();
        if planners.is_empty() {
            return;
        }
        while let Some(planner) = planners.pop() {
            state.planners.push_front(planner);
        }
        state.wake_all();
    }

    /// Publish a fetched morsel so any stream may read it.
    pub(super) fn publish(&self, morsel: PooledMorsel) {
        let mut state = self.inner.state.lock();
        state.morsels.push_back(morsel);
        state.wake_all();
    }

    /// Take the oldest published morsel, notifying its owner.
    pub(super) fn take_morsel(&self) -> Option<PooledMorsel> {
        let morsel = self.inner.state.lock().morsels.pop_front()?;
        morsel.owner.notify(morsel.unit, morsel.bytes);
        Some(morsel)
    }

    /// Is any published morsel waiting?
    pub(super) fn has_morsels(&self) -> bool {
        !self.inner.state.lock().morsels.is_empty()
    }

    /// Record that a file taken from this queue will not donate any more
    /// work (it produced its first morsel, donated, or finished).
    pub(super) fn settle(&self) {
        let mut state = self.inner.state.lock();
        state.planning = state.planning.saturating_sub(1);
        if state.planning == 0 {
            state.wake_all();
        }
    }
}
