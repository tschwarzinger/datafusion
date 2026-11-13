// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::VecDeque;
use std::sync::{Arc, OnceLock};

use crate::PartitionedFile;
use crate::file_groups::FileGroup;
use crate::file_scan_config::FileScanConfig;
use parking_lot::Mutex;

use super::work_pool::{PoolConfig, SharedPool, WorkPool};

/// Source of unopened files for `ScanState`.
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
    /// Pop the next unopened file.
    pub(super) fn pop_front(&mut self) -> Option<PartitionedFile> {
        match self {
            Self::Local(files) => files.pop_front(),
            Self::Shared(shared) => shared.pop_front(),
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
/// take the next unopened file from the front of the queue.
///
/// When the scan has an I/O budget, the siblings additionally share one
/// [`WorkPool`] holding all work that was opened but not yet read (planners,
/// in-flight I/O, fetched morsels), so any stream can drive any of it.
///
/// It uses a [`Mutex`] internally to provide thread-safe access
/// to the shared file queue.
#[derive(Debug, Clone)]
pub(crate) struct SharedWorkSource {
    inner: Arc<SharedWorkSourceInner>,
}

#[derive(Default)]
pub(super) struct SharedWorkSourceInner {
    files: Mutex<VecDeque<PartitionedFile>>,
    /// Work pool shared by all siblings (budgeted scans only).
    pool: OnceLock<SharedPool>,
}

impl std::fmt::Debug for SharedWorkSourceInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedWorkSourceInner")
            .field("files", &self.files)
            .field("pool", &self.pool.get().is_some())
            .finish()
    }
}

impl SharedWorkSource {
    /// Create a shared work source containing the provided unopened files.
    pub(crate) fn new(files: impl IntoIterator<Item = PartitionedFile>) -> Self {
        Self {
            inner: Arc::new(SharedWorkSourceInner {
                files: Mutex::new(files.into_iter().collect()),
                pool: OnceLock::new(),
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

    /// Pop the next file from the shared work queue.
    ///
    /// Returns `None` if the queue is empty
    fn pop_front(&self) -> Option<PartitionedFile> {
        self.inner.files.lock().pop_front()
    }

    /// The work pool shared by all siblings, created by the first caller.
    pub(super) fn pool(&self, config: PoolConfig) -> SharedPool {
        Arc::clone(self.inner.pool.get_or_init(|| WorkPool::new_shared(config)))
    }
}
