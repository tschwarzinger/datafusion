// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Structures for Morsel Driven IO.
//!
//! NOTE: As of DataFusion 54.0.0, these are experimental APIs that may change
//! substantially.
//!
//! Morsel Driven IO is a technique for parallelizing the reading of large files
//! by dividing them into smaller "morsels" that are processed independently.
//!
//! It is inspired by the paper [Morsel-Driven Parallelism: A NUMA-Aware Query
//! Evaluation Framework for the Many-Core Age](https://db.in.tum.de/~leis/papers/morsels.pdf).

mod adapters;
#[cfg(test)]
pub(crate) mod mocks;

use crate::PartitionedFile;
pub(crate) use adapters::FileOpenerMorselizer;
use arrow::array::RecordBatch;
use datafusion_common::Result;
use futures::FutureExt;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use std::fmt::Debug;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Return value of [`Morsel::into_stream`].
pub enum MorselStream {
    /// An `async` stream of [`RecordBatch`]es.
    ///
    /// This is used by morsels that still need to perform I/O while producing
    /// batches (for example, the [`FileOpener`] adapter). Implementations
    /// should prefer [`Self::Sync`] when all I/O has already completed so that
    /// I/O and CPU work are separated by the type system.
    ///
    /// [`FileOpener`]: crate::file_stream::FileOpener
    Async(BoxStream<'static, Result<RecordBatch>>),
    /// An iterator of [`RecordBatch`]es that does not require any more I/O.
    ///
    /// This API is explicitly not `async`: the morsel has all the data it
    /// needs and only performs CPU work (e.g. decoding) to produce batches.
    Sync(Box<dyn Iterator<Item = Result<RecordBatch>> + Send>),
}

impl Debug for MorselStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Async(_) => f.write_str("MorselStream::Async"),
            Self::Sync(_) => f.write_str("MorselStream::Sync"),
        }
    }
}

/// A Morsel of work ready to resolve to a stream of [`RecordBatch`]es.
///
/// This represents a single morsel of work that is ready to be processed. It
/// has all data necessary (does not need any I/O) and is ready to be turned
/// into a stream of [`RecordBatch`]es for processing by the execution engine.
pub trait Morsel: Send + Debug {
    /// Consume this morsel and produce a stream of [`RecordBatch`]es for processing.
    ///
    /// Note: This may do CPU work to decode already-loaded data, but should not
    /// do any I/O work such as reading from the file. Morsels that have all
    /// their data should return [`MorselStream::Sync`].
    fn into_stream(self: Box<Self>) -> Result<MorselStream>;

    /// Number of bytes of already-fetched (buffered) input this morsel holds
    /// until it has been fully consumed.
    ///
    /// This is used to bound the amount of fetched-but-not-yet-decoded data
    /// a scan keeps in memory when it issues I/O ahead of decoding (see
    /// `datafusion.execution.scan_io_budget_bytes`). Returns `0` by default.
    fn buffered_bytes(&self) -> usize {
        0
    }
}

/// A Morselizer takes a single [`PartitionedFile`] and creates the initial planner
/// for that file.
///
/// This is the entry point for morsel driven I/O.
pub trait Morselizer: Send + Sync + Debug {
    /// Return the initial [`MorselPlanner`] for this file.
    ///
    /// Morselizing a file may involve CPU work, such as parsing parquet
    /// metadata and evaluating pruning predicates. It should NOT do any I/O
    /// work, such as reading from the file. Any needed I/O should be done using
    /// [`MorselPlan::with_pending_planner`].
    fn plan_file(&self, file: PartitionedFile) -> Result<Box<dyn MorselPlanner>>;

    /// Allow (or disallow) this morselizer to split a file into several
    /// independent [`MorselPlanner`]s (returned together from a single
    /// [`MorselPlanner::plan`] call) whose morsels may be read in any order,
    /// concurrently, or by sibling streams.
    ///
    /// `FileStream` only enables this when the scan does not need to preserve
    /// order and has an I/O budget. The default implementation ignores it.
    fn set_split_files(&mut self, _split: bool) {}

    /// Does reading this morsel's output in planned order make the scan
    /// cheaper, even though any order is correct?
    ///
    /// For example, a Parquet scan whose row groups were reordered so that a
    /// TopK dynamic filter tightens as early as possible. `FileStream` then
    /// still prefetches (and may split / share) work, but consumes it in
    /// planned order. Defaults to `false`.
    fn prefers_ordered_read(&self) -> bool {
        false
    }
}

/// A Morsel Planner is responsible for creating morsels for a given scan.
///
/// The [`MorselPlanner`] is the unit of I/O. There is only ever a single I/O
/// outstanding for a specific planner. DataFusion may run
/// multiple planners in parallel, which corresponds to multiple parallel
/// I/O requests.
///
/// It is not a Rust `Stream` so that it can explicitly separate CPU bound
/// work from I/O work.
///
/// The design is similar to `ParquetPushDecoder`: when `plan` is called, it
/// should do CPU work to produce the next morsels or discover the next I/O
/// phase.
///
/// Best practice is to spawn I/O in a Tokio task on a separate runtime to
/// ensure that CPU work doesn't block or slow down I/O work, but this is not
/// strictly required by the API.
pub trait MorselPlanner: Send + Debug {
    /// Attempt to plan morsels. This may involve CPU work, such as parsing
    /// parquet metadata and evaluating pruning predicates.
    ///
    /// It should NOT do any I/O work, such as reading from the file. If I/O is
    /// required, the returned [`MorselPlan`] should contain a pending planner
    /// future that the caller polls to drive the I/O work to completion. Once
    /// that future resolves, it yields a planner ready for work.
    ///
    /// Note this function is **not async** to make it explicitly clear that if
    /// I/O is required, it should be done in the returned `io_future`.
    ///
    /// Returns `None` if the planner has no more work to do.
    ///
    /// # Empty Morsel Plans
    ///
    /// It may return `None`, which means no batches will be read from the file
    /// (e.g. due to late-pruning based on statistics).
    ///
    /// # Output Ordering
    ///
    /// See the comments on [`MorselPlan`] for the logical output order.
    fn plan(self: Box<Self>) -> Result<Option<MorselPlan>>;
}

/// Return result of [`MorselPlanner::plan`].
///
/// # Logical Ordering
///
/// For plans where the output order of rows is maintained, the output order of
/// a [`MorselPlanner`] is logically defined as follows:
/// 1. All morsels that are directly produced
/// 2. Recursively, all morsels produced by the returned `planners`
#[derive(Default)]
pub struct MorselPlan {
    /// Morsels ready for CPU work
    morsels: Vec<Box<dyn Morsel>>,
    /// Planners that are ready for CPU work.
    ready_planners: Vec<Box<dyn MorselPlanner>>,
    /// A future with planner I/O that resolves to a CPU ready planner.
    ///
    /// DataFusion will poll this future occasionally to drive the I/O work to
    /// completion. Once it resolves, planning continues with the returned
    /// planner.
    pending_planner: Option<PendingMorselPlanner>,
}

impl MorselPlan {
    /// Create an empty morsel plan.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the ready morsels.
    pub fn with_morsels(mut self, morsels: Vec<Box<dyn Morsel>>) -> Self {
        self.morsels = morsels;
        self
    }

    /// Set the ready child planners.
    pub fn with_planners(mut self, planners: Vec<Box<dyn MorselPlanner>>) -> Self {
        self.ready_planners = planners;
        self
    }

    /// Set the pending planner for an I/O phase.
    pub fn with_pending_planner<F>(mut self, io_future: F) -> Self
    where
        F: Future<Output = Result<Box<dyn MorselPlanner>>> + Send + 'static,
    {
        self.pending_planner = Some(PendingMorselPlanner::new(io_future));
        self
    }

    /// Set an already constructed pending planner (e.g. one carrying an
    /// [`PendingMorselPlanner::with_io_bytes`] hint).
    pub fn with_pending(mut self, pending: PendingMorselPlanner) -> Self {
        self.pending_planner = Some(pending);
        self
    }

    /// Set the pending planner  for an I/O phase.
    pub fn set_pending_planner<F>(&mut self, io_future: F)
    where
        F: Future<Output = Result<Box<dyn MorselPlanner>>> + Send + 'static,
    {
        self.pending_planner = Some(PendingMorselPlanner::new(io_future));
    }

    /// Take the ready morsels.
    pub fn take_morsels(&mut self) -> Vec<Box<dyn Morsel>> {
        std::mem::take(&mut self.morsels)
    }

    /// Take the ready child planners.
    pub fn take_ready_planners(&mut self) -> Vec<Box<dyn MorselPlanner>> {
        std::mem::take(&mut self.ready_planners)
    }

    /// Take the pending I/O future, if any.
    pub fn take_pending_planner(&mut self) -> Option<PendingMorselPlanner> {
        self.pending_planner.take()
    }

    /// Returns `true` if this plan contains an I/O future.
    pub fn has_io_future(&self) -> bool {
        self.pending_planner.is_some()
    }
}

/// Wrapper for I/O that must complete before planning can continue.
///
/// The wrapped future is lazy: no I/O is issued until it is first polled.
/// `FileStream` relies on this to only start I/O when it fits within its I/O
/// budget.
pub struct PendingMorselPlanner {
    future: BoxFuture<'static, Result<Box<dyn MorselPlanner>>>,
    /// Number of bytes this I/O will fetch, if known.
    io_bytes: Option<usize>,
}

impl PendingMorselPlanner {
    /// Create a new pending planner future.
    ///
    /// Example
    /// ```
    /// # use datafusion_common::DataFusionError;
    /// # use datafusion_datasource::morsel::{MorselPlanner, PendingMorselPlanner};
    /// let work = async move {
    ///  let planner: Box<dyn MorselPlanner> = {
    ///   // Do I/O work here, then return the next planner to run.
    ///  # unimplemented!();
    ///   };
    ///   Ok(planner) as Result<_, DataFusionError>;
    /// };
    /// let pending_io = PendingMorselPlanner::new(work);
    /// ```
    pub fn new<F>(future: F) -> Self
    where
        F: Future<Output = Result<Box<dyn MorselPlanner>>> + Send + 'static,
    {
        Self {
            future: future.boxed(),
            io_bytes: None,
        }
    }

    /// Record how many bytes this I/O is expected to fetch.
    ///
    /// When known, `FileStream` charges this amount against its I/O budget
    /// while the I/O is in flight. I/O of unknown size (e.g. metadata reads)
    /// is not charged, but `FileStream` will not start planning additional
    /// files while such I/O is outstanding.
    pub fn with_io_bytes(mut self, io_bytes: usize) -> Self {
        self.io_bytes = Some(io_bytes);
        self
    }

    /// Number of bytes this I/O is expected to fetch, if known.
    pub fn io_bytes(&self) -> Option<usize> {
        self.io_bytes
    }

    /// Consume this wrapper and return the underlying future.
    pub fn into_future(self) -> BoxFuture<'static, Result<Box<dyn MorselPlanner>>> {
        self.future
    }
}

/// Forwards polling to the underlying future.
impl Future for PendingMorselPlanner {
    type Output = Result<Box<dyn MorselPlanner>>;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // forward request to inner
        self.future.as_mut().poll(cx)
    }
}
