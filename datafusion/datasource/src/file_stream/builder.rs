// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use crate::file_scan_config::FileScanConfig;
use crate::file_stream::scan_state::{ReadOrder, ScanState};
use crate::file_stream::work_pool::PoolConfig;
use crate::file_stream::work_source::{SharedWorkSource, WorkSource};
use crate::morsel::{FileOpenerMorselizer, Morselizer};
use datafusion_common::{Result, internal_err};
use datafusion_execution::IoBudget;
use datafusion_physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet};

use super::metrics::FileStreamMetrics;
use super::{FileOpener, FileStream, FileStreamState, OnError};

/// Builder for constructing a [`FileStream`].
pub struct FileStreamBuilder<'a> {
    config: &'a FileScanConfig,
    partition: Option<usize>,
    morselizer: Option<Box<dyn Morselizer>>,
    metrics: Option<&'a ExecutionPlanMetricsSet>,
    on_error: OnError,
    shared_work_source: Option<SharedWorkSource>,
    io_budget: Option<Arc<IoBudget>>,
    split_morsels: bool,
}

impl<'a> FileStreamBuilder<'a> {
    /// Create a new builder for [`FileStream`].
    pub fn new(config: &'a FileScanConfig) -> Self {
        Self {
            config,
            partition: None,
            morselizer: None,
            metrics: None,
            on_error: OnError::Fail,
            shared_work_source: None,
            io_budget: None,
            split_morsels: true,
        }
    }

    /// Configure the partition to scan.
    pub fn with_partition(mut self, partition: usize) -> Self {
        self.partition = Some(partition);
        self
    }

    /// Configure the [`FileOpener`] used to open files.
    ///
    /// This will overwrite any setting from [`Self::with_morselizer`]
    pub fn with_file_opener(mut self, file_opener: Arc<dyn FileOpener>) -> Self {
        self.morselizer = Some(Box::new(FileOpenerMorselizer::new(file_opener)));
        self
    }

    /// Configure the [`Morselizer`] used to open files.
    ///
    /// This will overwrite any setting from [`Self::with_file_opener`]
    pub fn with_morselizer(mut self, morselizer: Box<dyn Morselizer>) -> Self {
        self.morselizer = Some(morselizer);
        self
    }

    /// Configure the metrics set used by the stream.
    pub fn with_metrics(mut self, metrics: &'a ExecutionPlanMetricsSet) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Configure the behavior when opening or scanning a file fails.
    pub fn with_on_error(mut self, on_error: OnError) -> Self {
        self.on_error = on_error;
        self
    }

    /// Configure the [`SharedWorkSource`] for sibling work stealing.
    pub(crate) fn with_shared_work_source(
        mut self,
        shared_work_source: Option<SharedWorkSource>,
    ) -> Self {
        self.shared_work_source = shared_work_source;
        self
    }

    /// Configure the query-wide I/O budget used to prefetch I/O ahead of
    /// decoding (see [`TaskContext::io_budget`]). Without one (the default),
    /// at most one I/O is outstanding at a time.
    ///
    /// [`TaskContext::io_budget`]: datafusion_execution::TaskContext::io_budget
    pub fn with_io_budget(mut self, io_budget: Option<Arc<IoBudget>>) -> Self {
        self.io_budget = io_budget;
        self
    }

    /// Configure whether files may be split into independent morsels
    /// (see `datafusion.execution.scan_split_morsels`).
    pub fn with_split_morsels(mut self, split_morsels: bool) -> Self {
        self.split_morsels = split_morsels;
        self
    }

    /// Build the configured [`FileStream`].
    pub fn build(self) -> Result<FileStream> {
        let Self {
            config,
            partition,
            morselizer,
            metrics,
            on_error,
            shared_work_source,
            io_budget,
            split_morsels,
        } = self;

        let Some(partition) = partition else {
            return internal_err!("FileStreamBuilder missing required partition");
        };
        let Some(mut morselizer) = morselizer else {
            return internal_err!("FileStreamBuilder missing required morselizer");
        };
        let Some(metrics) = metrics else {
            return internal_err!("FileStreamBuilder missing required metrics");
        };
        let projected_schema = config.projected_schema()?;
        let Some(file_group) = config.file_groups.get(partition).cloned() else {
            return internal_err!(
                "FileStreamBuilder invalid partition index: {partition}"
            );
        };
        let work_source = match shared_work_source {
            Some(shared) => WorkSource::Shared(shared),
            None => WorkSource::Local(file_group.into_inner().into()),
        };

        let preserve_order = config.preserves_file_order();
        let io_budget =
            io_budget.filter(|budget| budget.is_enabled() && config.limit.is_none());
        let split = split_morsels && io_budget.is_some() && !preserve_order;
        morselizer.set_split_files(split);
        let read_order = if preserve_order {
            ReadOrder::Required
        } else if morselizer.prefers_ordered_read() {
            ReadOrder::Preferred
        } else {
            ReadOrder::Any
        };

        let file_stream_metrics = FileStreamMetrics::new(metrics, partition);
        let scan_state = Box::new(ScanState::new(
            work_source,
            config.limit,
            morselizer,
            on_error,
            file_stream_metrics,
            PoolConfig {
                io_budget,
                read_order,
                streams: config.file_groups.len(),
            },
        ));

        Ok(FileStream {
            projected_schema,
            state: FileStreamState::Scan { scan_state },
            baseline_metrics: BaselineMetrics::new(metrics, partition),
        })
    }
}
