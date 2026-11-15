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

    /// Scheduling information: [`WorkInfo::bytes`] is the fetched data this
    /// morsel holds until it has been consumed. Unknown by default.
    fn info(&self) -> WorkInfo {
        WorkInfo::default()
    }
}

/// Scheduling information about an item of morsel work (a pending I/O, a
/// planner or a morsel), which `FileStream` uses to bound and order work it
/// does ahead of time (see `datafusion.execution.scan_io_budget_bytes`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WorkInfo {
    /// The data the item accounts for, if known: for a pending I/O the bytes
    /// it will fetch, for a morsel the fetched bytes it holds until consumed.
    ///
    /// `FileStream` charges known sizes against the query's I/O budget. Work
    /// of unknown size is not charged, but limits how much further work is
    /// started (e.g. no further files are opened in bursts while metadata of
    /// unknown size is being read).
    pub bytes: Option<usize>,
    /// Planners only: plan this planner only when its output is needed (a
    /// stream has nothing to read), rather than eagerly when it is ready.
    ///
    /// For planners whose next CPU step makes decisions that improve as the
    /// query progresses. For example, a Parquet decoder evaluates a pushed
    /// down row filter in the step after the filter columns arrive, and a
    /// TopK dynamic filter in that predicate prunes far more rows once TopK
    /// has seen earlier rows. Any I/O is still prefetched.
    pub defer: bool,
}

impl WorkInfo {
    /// Information about an item of `bytes` bytes.
    pub fn with_bytes(bytes: usize) -> Self {
        Self {
            bytes: Some(bytes),
            defer: false,
        }
    }

    /// Set [`Self::defer`].
    pub fn with_defer(mut self, defer: bool) -> Self {
        self.defer = defer;
        self
    }
}

/// How `FileStream` will consume a [`Morselizer`]'s output, see
/// [`Morselizer::configure`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MorselizerOptions {
    /// The morselizer may split a file into several independent
    /// [`MorselPlanner`]s, returned together from a single
    /// [`MorselPlanner::plan`] call in the file's logical order.
    ///
    /// `FileStream` may then fetch the parts concurrently. When the scan must
    /// preserve order (or the morselizer prefers it, see
    /// [`MorselizerProperties::prefers_ordered_read`]) it reads them in the
    /// order they were returned; otherwise in any order and in any sibling
    /// stream.
    pub split_files: bool,
}

/// Properties of a [`Morselizer`]'s output, see [`Morselizer::configure`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MorselizerProperties {
    /// Reading the output in planned order makes the scan cheaper, even
    /// though any order is correct: for example a Parquet scan whose row
    /// groups were reordered so that a TopK dynamic filter tightens as early
    /// as possible. `FileStream` then still prefetches (and may split and
    /// share) work, but consumes it in planned order.
    pub prefers_ordered_read: bool,
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

    /// Tell the morselizer how its output will be consumed and get its
    /// properties. Called once by `FileStream` before any file is planned.
    /// The default ignores the options and has default properties.
    fn configure(&mut self, _options: &MorselizerOptions) -> MorselizerProperties {
        MorselizerProperties::default()
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

    /// Scheduling information, see [`WorkInfo::defer`]. `FileStream`
    /// otherwise plans ready planners eagerly. Default: not deferred.
    fn info(&self) -> WorkInfo {
        WorkInfo::default()
    }
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
    /// [`PendingMorselPlanner::with_info`]).
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
    /// Scheduling information, see [`WorkInfo::bytes`].
    info: WorkInfo,
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
            info: WorkInfo::default(),
        }
    }

    /// Set the scheduling information, e.g. how many bytes this I/O will
    /// fetch ([`WorkInfo::with_bytes`]).
    pub fn with_info(mut self, info: WorkInfo) -> Self {
        self.info = info;
        self
    }

    /// Scheduling information.
    pub fn info(&self) -> WorkInfo {
        self.info
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
