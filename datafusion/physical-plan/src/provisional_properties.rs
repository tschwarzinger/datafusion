// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! SPIKE: *provisional* plan properties.
//!
//! An operator's [`PlanProperties`] describe its output **assuming its input
//! requirements are met**. Before `EnforceDistribution` / `EnforceSorting`
//! have run, that may not be the case. For example, a [`SortExec`] with
//! `preserve_partitioning = false` requires a single input partition. Until
//! the optimizer makes that true, the ordering and equivalences it reports
//! can be derived from per-partition facts of a multi-partition input, and so
//! can be wrong.
//!
//! This module lets an operator mark its properties as *provisional* and
//! gives callers a typed error, [`UnmetInputRequirements`], that can be told
//! apart from other errors with [`is_unmet_input_requirements`].
//!
//! It also has opt-in instrumentation for the existing infallible accessors,
//! [`PlanProperties::equivalence_properties`] and
//! [`PlanProperties::output_ordering`]. This estimates how much code would
//! break if those accessors returned errors. Set the environment variable
//! `DATAFUSION_SPIKE_PROVISIONAL` to:
//!
//! * `record`: append one line per read of provisional properties to the file
//!   named by `DATAFUSION_SPIKE_PROVISIONAL_LOG` (default:
//!   `$TMPDIR/datafusion_provisional_reads.log`).
//! * `panic`: panic on the first such read.
//!
//! When unset (the default), the accessors behave exactly as before.
//!
//! [`PlanProperties`]: crate::PlanProperties
//! [`PlanProperties::equivalence_properties`]: crate::PlanProperties::equivalence_properties
//! [`PlanProperties::output_ordering`]: crate::PlanProperties::output_ordering
//! [`SortExec`]: crate::sorts::sort::SortExec

use std::backtrace::Backtrace;
use std::fmt;
use std::io::Write;
use std::panic::Location;
use std::path::PathBuf;
use std::sync::OnceLock;

use datafusion_common::DataFusionError;

/// Error returned when a plan property is requested before the operator's
/// input requirements are met (i.e. before `EnforceDistribution` /
/// `EnforceSorting` have run, or for a plan they have not fixed up).
///
/// Use [`is_unmet_input_requirements`] to recognize it in a
/// [`DataFusionError`] chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnmetInputRequirements {
    /// Name of the operator whose input requirements are unmet.
    pub operator: &'static str,
    /// Human-readable description of the unmet requirement.
    pub requirement: String,
}

impl UnmetInputRequirements {
    /// Create a new [`UnmetInputRequirements`].
    pub fn new(operator: &'static str, requirement: impl Into<String>) -> Self {
        Self {
            operator,
            requirement: requirement.into(),
        }
    }
}

impl fmt::Display for UnmetInputRequirements {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "plan properties of {} are provisional: {}",
            self.operator, self.requirement
        )
    }
}

impl std::error::Error for UnmetInputRequirements {}

impl From<UnmetInputRequirements> for DataFusionError {
    fn from(value: UnmetInputRequirements) -> Self {
        DataFusionError::External(Box::new(value))
    }
}

/// Returns `true` if `error`, or any error in its source chain, is an
/// [`UnmetInputRequirements`].
pub fn is_unmet_input_requirements(error: &DataFusionError) -> bool {
    let mut current: Option<&(dyn std::error::Error + 'static)> = Some(error);
    while let Some(e) = current {
        if e.is::<UnmetInputRequirements>() {
            return true;
        }
        if let Some(DataFusionError::External(inner)) =
            e.downcast_ref::<DataFusionError>()
            && inner.is::<UnmetInputRequirements>()
        {
            return true;
        }
        current = e.source();
    }
    false
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Off,
    Record,
    Panic,
}

fn mode() -> Mode {
    static MODE: OnceLock<Mode> = OnceLock::new();
    *MODE.get_or_init(
        || match std::env::var("DATAFUSION_SPIKE_PROVISIONAL").as_deref() {
            Ok("record") => Mode::Record,
            Ok("panic") => Mode::Panic,
            _ => Mode::Off,
        },
    )
}

fn log_path() -> &'static PathBuf {
    static PATH: OnceLock<PathBuf> = OnceLock::new();
    PATH.get_or_init(
        || match std::env::var_os("DATAFUSION_SPIKE_PROVISIONAL_LOG") {
            Some(path) => PathBuf::from(path),
            None => std::env::temp_dir().join("datafusion_provisional_reads.log"),
        },
    )
}

/// Called by the infallible property accessors when they read provisional
/// properties. A no-op unless `DATAFUSION_SPIKE_PROVISIONAL` is set.
#[track_caller]
#[inline]
pub(crate) fn on_provisional_read(
    accessor: &'static str,
    reason: &UnmetInputRequirements,
) {
    match mode() {
        Mode::Off => {}
        Mode::Panic => panic!(
            "{accessor} read provisional properties at {}: {reason}",
            Location::caller()
        ),
        Mode::Record => record(accessor, Location::caller(), reason),
    }
}

#[cold]
fn record(
    accessor: &'static str,
    location: &Location<'_>,
    reason: &UnmetInputRequirements,
) {
    let backtrace = Backtrace::force_capture();
    let frames = if std::env::var_os("DATAFUSION_SPIKE_PROVISIONAL_RAW").is_some() {
        backtrace.to_string().replace('\n', " | ")
    } else {
        datafusion_frames(&backtrace, 12).join(" <- ")
    };
    let thread = std::thread::current();
    let line = format!(
        "{accessor}\t{}:{}\t{}\t{}\t{frames}\n",
        location.file(),
        location.line(),
        reason.operator,
        thread.name().unwrap_or("<unnamed>"),
    );
    // One `write_all` per line on an `O_APPEND` file keeps lines from
    // concurrent test threads and processes from interleaving in practice.
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path())
    {
        let _ = file.write_all(line.as_bytes());
    }
}

/// Extract up to `limit` DataFusion function names from a backtrace,
/// innermost first, skipping this module and the accessors themselves.
fn datafusion_frames(backtrace: &Backtrace, limit: usize) -> Vec<String> {
    backtrace
        .to_string()
        .lines()
        .filter_map(|line| {
            // Frame lines look like `  12: datafusion_physical_plan::foo::bar`
            let (index, name) = line.trim_start().split_once(": ")?;
            index.parse::<usize>().ok()?;
            Some(name.trim())
        })
        .filter(|name| name.starts_with("datafusion") || name.starts_with("<datafusion"))
        .filter(|name| {
            !name.contains("provisional_properties")
                && !name.contains("PlanProperties>::equivalence_properties")
                && !name.contains("PlanProperties>::output_ordering")
                && !name.contains("ExecutionPlanProperties")
        })
        .map(|name| {
            // Drop the trailing hash, e.g. `::h0123456789abcdef`.
            match name.rsplit_once("::h") {
                Some((prefix, hash))
                    if hash.len() == 16
                        && hash.chars().all(|c| c.is_ascii_hexdigit()) =>
                {
                    prefix.to_string()
                }
                _ => name.to_string(),
            }
        })
        .take(limit)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distinguishes_unmet_requirements_from_other_errors() {
        let unmet: DataFusionError =
            UnmetInputRequirements::new("SortExec", "requires a single input partition")
                .into();
        assert!(is_unmet_input_requirements(&unmet));

        let wrapped = unmet.context("while planning COPY");
        assert!(is_unmet_input_requirements(&wrapped));

        let other = DataFusionError::Plan("something else".to_string());
        assert!(!is_unmet_input_requirements(&other));
        assert!(!is_unmet_input_requirements(&other.context("ctx")));
    }
}
