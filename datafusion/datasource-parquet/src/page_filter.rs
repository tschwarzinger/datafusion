// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Contains code to filter entire pages

use std::collections::HashSet;
use std::sync::Arc;

use super::metrics::ParquetFileMetrics;
use crate::metadata::has_untrusted_min_max_order;
use crate::pruning::build_inverted_predicate;
use crate::{ParquetAccessPlan, RowGroupAccess};

use arrow::array::BooleanArray;
use arrow::{
    array::ArrayRef,
    datatypes::{Schema, SchemaRef},
};
use datafusion_common::ScalarValue;
use datafusion_common::pruning::PruningStatistics;
use datafusion_physical_expr::{PhysicalExpr, split_conjunction};
use datafusion_pruning::{MAX_IN_LIST_SIZE, PruningPredicate, PruningPredicateBuilder};

use log::{debug, trace};
use parquet::arrow::arrow_reader::statistics::StatisticsConverter;
use parquet::file::metadata::page_index::PageIndexProvider;
use parquet::file::page_index::offset_index::PageLocation;
use parquet::schema::types::SchemaDescriptor;
use parquet::{
    arrow::arrow_reader::{RowSelection, RowSelector},
    file::metadata::{ParquetMetaData, RowGroupMetaData},
};

/// Filters a [`ParquetAccessPlan`] based on the [Parquet PageIndex], if present
///
/// It does so by evaluating statistics from the [`PageIndexProvider`] (column
/// and offset indexes) and converting them to [`RowSelection`].
///
/// [Parquet PageIndex]: https://github.com/apache/parquet-format/blob/master/PageIndex.md
///
/// For example, given a row group with two column (chunks) for `A`
/// and `B` with the following with page level statistics:
///
/// ```text
/// ┏━━ ━━━ ━━━ ━━━ ━━━ ━━━ ━━━ ━━━ ━━━ ━━━ ━━━ ━━━ ━━━
///    ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─   ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─   ┃
/// ┃     ┌──────────────┐  │     ┌──────────────┐  │  ┃
/// ┃  │  │              │     │  │              │     ┃
/// ┃     │              │  │     │     Page     │  │
///    │  │              │     │  │      3       │     ┃
/// ┃     │              │  │     │   min: "A"   │  │  ┃
/// ┃  │  │              │     │  │   max: "C"   │     ┃
/// ┃     │     Page     │  │     │ first_row: 0 │  │
///    │  │      1       │     │  │              │     ┃
/// ┃     │   min: 10    │  │     └──────────────┘  │  ┃
/// ┃  │  │   max: 20    │     │  ┌──────────────┐     ┃
/// ┃     │ first_row: 0 │  │     │              │  │
///    │  │              │     │  │     Page     │     ┃
/// ┃     │              │  │     │      4       │  │  ┃
/// ┃  │  │              │     │  │   min: "D"   │     ┃
/// ┃     │              │  │     │   max: "G"   │  │
///    │  │              │     │  │first_row: 100│     ┃
/// ┃     └──────────────┘  │     │              │  │  ┃
/// ┃  │  ┌──────────────┐     │  │              │     ┃
/// ┃     │              │  │     └──────────────┘  │
///    │  │     Page     │     │  ┌──────────────┐     ┃
/// ┃     │      2       │  │     │              │  │  ┃
/// ┃  │  │   min: 30    │     │  │     Page     │     ┃
/// ┃     │   max: 40    │  │     │      5       │  │
///    │  │first_row: 200│     │  │   min: "H"   │     ┃
/// ┃     │              │  │     │   max: "Z"   │  │  ┃
/// ┃  │  │              │     │  │first_row: 250│     ┃
/// ┃     └──────────────┘  │     │              │  │
///    │                       │  └──────────────┘     ┃
/// ┃   ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘   ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘  ┃
/// ┃       ColumnChunk            ColumnChunk         ┃
/// ┃            A                      B
///  ━━━ ━━━ ━━━ ━━━ ━━━ ━━━ ━━━ ━━━ ━━━ ━━━ ━━━ ━━━ ━━┛
///
///   Total rows: 300
/// ```
///
/// Given the predicate `A > 35 AND B = 'F'`:
///
/// Using `A > 35`: can rule out all of values in Page 1 (rows 0 -> 199)
///
/// Using `B = 'F'`: can rule out all values in Page 3 and Page 5 (rows 0 -> 99, and 250 -> 299)
///
/// So we can entirely skip rows 0->199 and 250->299 as we know they
/// can not contain rows that match the predicate.
///
/// # Implementation notes
///
/// Single column predicates are evaluated using the PageIndex information
/// for that column to determine which row ranges can be skipped based.
///
/// The resulting [`RowSelection`]'s are combined into a final
/// row selection that is added to the [`ParquetAccessPlan`].
#[derive(Debug)]
pub struct PagePruningAccessPlanFilter {
    /// single column predicates (e.g. (`col = 5`) extracted from the overall
    /// predicate. Must all be true for a row to be included in the result.
    predicates: Vec<PruningPredicate>,
    /// Whether every original conjunct can be evaluated from one column's page
    /// statistics. Otherwise no range can be classified as fully matched.
    all_predicates_supported: bool,
}

/// Result of applying page-index pruning to a [`ParquetAccessPlan`].
pub(crate) struct PagePruningResult {
    pub(crate) access_plan: ParquetAccessPlan,
    /// Pages skipped because the containing row group was fully matched by
    /// row-group statistics.
    pub(crate) pages_skipped_by_fully_matched: usize,
    /// Rows skipped by page-level limit pruning.
    pub(crate) limit_pruned_rows: usize,
}

impl PagePruningResult {
    fn new(
        access_plan: ParquetAccessPlan,
        pages_skipped_by_fully_matched: usize,
        limit_pruned_rows: usize,
    ) -> Self {
        Self {
            access_plan,
            pages_skipped_by_fully_matched,
            limit_pruned_rows,
        }
    }
}

impl PagePruningAccessPlanFilter {
    /// Create a new [`PagePruningAccessPlanFilter`] from a physical
    /// expression, using the default `IN (...)` pruning limit.
    #[expect(clippy::needless_pass_by_value)]
    pub fn new(expr: &Arc<dyn PhysicalExpr>, schema: SchemaRef) -> Self {
        Self::new_with_max_in_list_size(expr, &schema, MAX_IN_LIST_SIZE)
    }

    /// Create a page filter using the same `IN (...)` limit as row-group pruning.
    pub(crate) fn new_with_max_in_list_size(
        expr: &Arc<dyn PhysicalExpr>,
        schema: &SchemaRef,
        max_in_list_size: usize,
    ) -> Self {
        let mut predicates = vec![];
        let mut all_predicates_supported = true;

        // Ordinary pruning can use any supported conjunct. Fully matched
        // ranges require every original conjunct to be supported.
        for predicate in split_conjunction(expr) {
            let pp = match PruningPredicateBuilder::new()
                .with_file_schema(Arc::clone(schema))
                .with_max_in_list_size(max_in_list_size)
                .try_build(Arc::clone(predicate))
            {
                Ok(pp) => pp,
                Err(e) => {
                    debug!("Ignoring error creating page pruning predicate: {e}");
                    all_predicates_supported = false;
                    continue;
                }
            };

            if pp.always_true() {
                debug!("Ignoring always true page pruning predicate: {predicate}");
                all_predicates_supported = false;
                continue;
            }

            if pp.required_columns().single_column().is_none() {
                debug!("Ignoring multi-column page pruning predicate: {predicate}");
                all_predicates_supported = false;
                continue;
            }

            predicates.push(pp);
        }

        Self {
            predicates,
            all_predicates_supported,
        }
    }

    /// Returns an updated [`ParquetAccessPlan`] by applying predicates to the
    /// parquet page index, if any
    pub fn prune_plan_with_page_index(
        &self,
        access_plan: ParquetAccessPlan,
        arrow_schema: &Schema,
        parquet_schema: &SchemaDescriptor,
        parquet_metadata: &ParquetMetaData,
        file_metrics: &ParquetFileMetrics,
    ) -> ParquetAccessPlan {
        self.prune_plan_with_page_index_and_metrics(
            access_plan,
            arrow_schema,
            parquet_schema,
            parquet_metadata,
            file_metrics,
            None,
        )
        .access_plan
    }

    /// Returns an updated [`ParquetAccessPlan`] and metrics by applying predicates
    /// to the parquet page index, if any.
    pub(crate) fn prune_plan_with_page_index_and_metrics(
        &self,
        mut access_plan: ParquetAccessPlan,
        arrow_schema: &Schema,
        parquet_schema: &SchemaDescriptor,
        parquet_metadata: &ParquetMetaData,
        file_metrics: &ParquetFileMetrics,
        limit: Option<usize>,
    ) -> PagePruningResult {
        // scoped timer updates on drop
        let _timer_guard = file_metrics.page_index_eval_time.timer();
        if self.predicates.is_empty() {
            return PagePruningResult::new(access_plan, 0, 0);
        }

        let page_index_predicates = &self.predicates;
        let groups = parquet_metadata.row_groups();

        if groups.is_empty() {
            return PagePruningResult::new(access_plan, 0, 0);
        }

        if !parquet_metadata
            .page_index()
            .is_some_and(|page_index| page_index.is_complete())
        {
            debug!(
                "Can not prune pages due to lack of indexes. Have offset: {}, column index: {}",
                parquet_metadata
                    .page_index()
                    .is_some_and(|page_index| page_index.has_offset_indexes()),
                parquet_metadata
                    .page_index()
                    .is_some_and(|page_index| page_index.has_column_indexes())
            );
            return PagePruningResult::new(access_plan, 0, 0);
        }

        // Build inverse predicates only for unordered LIMIT queries. Ordinary
        // page pruning does not need them.
        let inverted_predicates: Option<Vec<_>> =
            if limit.is_some() && self.all_predicates_supported {
                self.predicates
                    .iter()
                    .map(|predicate| {
                        let inverted = build_inverted_predicate(predicate, arrow_schema)?;
                        inverted
                            .required_columns()
                            .single_column()
                            .is_some()
                            .then_some(inverted)
                    })
                    .collect()
            } else {
                None
            };

        // track the total number of rows that should be skipped
        let mut total_skip = 0;
        // track the total number of rows that should not be skipped
        let mut total_select = 0;
        // track the total number of pages that should be skipped
        let mut total_pages_skip = 0;
        // track the total number of pages that should not be skipped
        let mut total_pages_select = 0;
        // track pages for which page-index pruning was skipped because the
        // containing row group was already proven fully matched by statistics
        let mut total_pages_skipped_by_fully_matched = 0;
        let mut fully_matched_page_selections = inverted_predicates
            .as_ref()
            .map(|_| vec![None; groups.len()]);
        let mut guaranteed_rows = 0;

        // for each row group specified in the access plan
        let row_group_indexes = access_plan.row_group_indexes();
        // Evaluate each predicate once for the pages of all row groups that are not
        // fully matched, instead of once per row group.
        let pruned_row_groups = row_group_indexes
            .iter()
            .copied()
            .filter(|&idx| !access_plan.is_fully_matched(idx))
            .collect::<Vec<_>>();
        let pruned_pages = prune_pages_in_row_groups(
            &pruned_row_groups,
            page_index_predicates,
            arrow_schema,
            parquet_schema,
            parquet_metadata,
            file_metrics,
        );
        let mut pruned_pages = pruned_row_groups.into_iter().zip(pruned_pages);
        for row_group_index in row_group_indexes {
            // Skip page pruning for fully matched row groups: all rows are
            // known to satisfy the predicate, so page-level pruning is wasted work.
            if access_plan.is_fully_matched(row_group_index) {
                let page_count =
                    fully_matched_page_count(row_group_index, parquet_metadata);
                total_pages_skipped_by_fully_matched += page_count;
                guaranteed_rows += selected_rows_in_access(
                    &access_plan.inner()[row_group_index],
                    &groups[row_group_index],
                );

                continue;
            }
            let (
                pruned_row_group,
                RowGroupPages {
                    overall_selection,
                    matched_pages,
                },
            ) = pruned_pages
                .next()
                .expect("a pruning result for every row group that is not fully matched");
            debug_assert_eq!(pruned_row_group, row_group_index);

            if let Some(overall_selection) = overall_selection {
                let rows_selected = overall_selection.row_count();
                if rows_selected > 0 {
                    let rows_skipped = overall_selection.skipped_row_count();
                    trace!(
                        "Overall selection from predicate skipped {rows_skipped}, selected {rows_selected}: {overall_selection:?}"
                    );
                    total_skip += rows_skipped;
                    total_select += rows_selected;
                    access_plan.scan_selection(row_group_index, overall_selection)
                } else {
                    // Selection skips all rows, so skip the entire row group
                    let rows_skipped = groups[row_group_index].num_rows() as usize;
                    access_plan.skip(row_group_index);
                    total_skip += rows_skipped;
                    trace!(
                        "Overall selection from predicate is empty, \
                        skipping all {rows_skipped} rows in row group {row_group_index}"
                    );
                }
            } else {
                total_select +=
                    parquet_metadata.row_group(row_group_index).num_rows() as usize;
            }

            if guaranteed_rows < limit.unwrap_or(0)
                && access_plan.should_scan(row_group_index)
                && let Some(inverted_predicates) = &inverted_predicates
            {
                let mut fully_matched_selection = None;
                let mut complete = true;

                for predicate in inverted_predicates {
                    let Some((selection, _)) = prune_pages_for_predicate(
                        row_group_index,
                        predicate,
                        arrow_schema,
                        parquet_schema,
                        parquet_metadata,
                        file_metrics,
                    ) else {
                        complete = false;
                        break;
                    };

                    fully_matched_selection = update_selection(
                        fully_matched_selection,
                        complement_selection(selection),
                    );

                    // An empty intersection cannot gain rows from later conjuncts.
                    if !fully_matched_selection
                        .as_ref()
                        .is_some_and(|selection| selection.selects_any())
                    {
                        break;
                    }
                }

                if complete
                    && let Some(selections) = fully_matched_page_selections.as_mut()
                    && let Some(selection) = fully_matched_selection
                {
                    guaranteed_rows += match &access_plan.inner()[row_group_index] {
                        RowGroupAccess::Skip => 0,
                        RowGroupAccess::Scan => selection.row_count(),
                        RowGroupAccess::Selection(existing) => {
                            existing.intersection(&selection).row_count()
                        }
                    };
                    selections[row_group_index] = Some(selection);
                }
            }

            let pages_matched = matched_pages.iter().filter(|matched| **matched).count();
            total_pages_select += pages_matched;
            total_pages_skip += matched_pages.len() - pages_matched;
        }

        file_metrics.page_index_rows_pruned.add_pruned(total_skip);
        file_metrics
            .page_index_rows_pruned
            .add_matched(total_select);
        file_metrics
            .page_index_pages_pruned
            .add_pruned(total_pages_skip);
        file_metrics
            .page_index_pages_pruned
            .add_matched(total_pages_select);

        let mut limit_pruned_rows = 0;
        if let Some(limit) = limit
            && let Some(fully_matched_page_selections) =
                fully_matched_page_selections.as_deref()
            && let Some((limit_plan, pruned_rows, pruned_row_groups)) = limit_pruned_plan(
                &access_plan,
                fully_matched_page_selections,
                limit,
                groups,
            )
        {
            access_plan = limit_plan;
            limit_pruned_rows = pruned_rows;
            file_metrics
                .limit_pruned_row_groups
                .add_pruned(pruned_row_groups);
        }

        PagePruningResult::new(
            access_plan,
            total_pages_skipped_by_fully_matched,
            limit_pruned_rows,
        )
    }

    /// Returns the number of filters in the [`PagePruningAccessPlanFilter`]
    pub fn filter_number(&self) -> usize {
        self.predicates.len()
    }

    /// Returns the names of the columns referenced by the page pruning
    /// predicates (each predicate references exactly one column, see
    /// [`Self::new`]).
    pub(crate) fn predicate_column_names(&self) -> impl Iterator<Item = &str> {
        self.predicates
            .iter()
            .filter_map(|p| p.required_columns().single_column())
            .map(|c| c.name())
    }
}

fn update_selection(
    current_selection: Option<RowSelection>,
    row_selection: RowSelection,
) -> Option<RowSelection> {
    match current_selection {
        None => Some(row_selection),
        Some(current_selection) => Some(current_selection.intersection(&row_selection)),
    }
}

fn complement_selection(selection: RowSelection) -> RowSelection {
    let selectors: Vec<RowSelector> = selection.into();
    selectors
        .into_iter()
        .map(|selector| {
            if selector.skip {
                RowSelector::select(selector.row_count)
            } else {
                RowSelector::skip(selector.row_count)
            }
        })
        .collect()
}

/// Build a plan containing only ranges known to satisfy the predicate, if
/// those ranges contain enough rows to satisfy `limit`.
fn limit_pruned_plan(
    access_plan: &ParquetAccessPlan,
    fully_matched_page_selections: &[Option<RowSelection>],
    limit: usize,
    groups: &[RowGroupMetaData],
) -> Option<(ParquetAccessPlan, usize, usize)> {
    let original_rows = selected_row_count(access_plan, groups);
    let original_row_groups = access_plan.row_group_indexes().len();
    let mut candidate_plan = ParquetAccessPlan::new_none(access_plan.len());
    if limit == 0 {
        return Some((candidate_plan, original_rows, original_row_groups));
    }

    let mut candidate_rows = 0;

    for row_group_index in access_plan.row_group_indexes() {
        let access = if access_plan.is_fully_matched(row_group_index) {
            access_plan.inner()[row_group_index].clone()
        } else {
            let Some(selection) = fully_matched_page_selections[row_group_index].as_ref()
            else {
                continue;
            };
            match &access_plan.inner()[row_group_index] {
                RowGroupAccess::Skip => continue,
                RowGroupAccess::Scan => RowGroupAccess::Selection(selection.clone()),
                RowGroupAccess::Selection(existing) => {
                    RowGroupAccess::Selection(existing.intersection(selection))
                }
            }
        };

        let rows = selected_rows_in_access(&access, &groups[row_group_index]);
        if rows == 0 {
            continue;
        }

        candidate_plan.set(row_group_index, access);
        if access_plan.is_fully_matched(row_group_index) {
            candidate_plan.mark_fully_matched(row_group_index);
        }
        candidate_rows += rows;

        if candidate_rows >= limit {
            let pruned_rows = original_rows.saturating_sub(candidate_rows);
            let pruned_row_groups = original_row_groups
                .saturating_sub(candidate_plan.row_group_indexes().len());
            return Some((candidate_plan, pruned_rows, pruned_row_groups));
        }
    }

    None
}

fn selected_row_count(
    access_plan: &ParquetAccessPlan,
    groups: &[RowGroupMetaData],
) -> usize {
    access_plan
        .inner()
        .iter()
        .zip(groups)
        .map(|(access, group)| selected_rows_in_access(access, group))
        .sum()
}

fn selected_rows_in_access(access: &RowGroupAccess, group: &RowGroupMetaData) -> usize {
    match access {
        RowGroupAccess::Skip => 0,
        RowGroupAccess::Scan => group.num_rows() as usize,
        RowGroupAccess::Selection(selection) => selection.row_count(),
    }
}

/// Returns the number of pages for which page-index pruning is skipped because
/// the containing row group is fully matched by row-group statistics.
fn fully_matched_page_count(
    row_group_index: usize,
    parquet_metadata: &ParquetMetaData,
) -> usize {
    parquet_metadata
        .page_index()
        .and_then(|page_index| page_index.offset_index(row_group_index, 0))
        .map_or(0, |offset_index| offset_index.page_locations().len())
}

/// The result of pruning the pages of a row group with all predicates.
struct RowGroupPages {
    /// The intersection of the selections of all predicates, or `None` if no
    /// predicate could be evaluated.
    overall_selection: Option<RowSelection>,
    /// Whether each page (of the first column) may match all predicates.
    matched_pages: Vec<bool>,
}

/// Prunes the pages of the given row groups with all `predicates`.
///
/// Each predicate is evaluated once for the pages of all row groups whose
/// selection is not empty yet.
fn prune_pages_in_row_groups(
    row_groups: &[usize],
    predicates: &[PruningPredicate],
    arrow_schema: &Schema,
    parquet_schema: &SchemaDescriptor,
    parquet_metadata: &ParquetMetaData,
    metrics: &ParquetFileMetrics,
) -> Vec<RowGroupPages> {
    let mut results = row_groups
        .iter()
        .map(|&row_group_index| {
            let total_pages_in_group = parquet_metadata
                .page_index()
                .and_then(|page_index| page_index.offset_index(row_group_index, 0))
                .map_or(0, |offset_index| offset_index.page_locations().len());
            RowGroupPages {
                overall_selection: None,
                matched_pages: vec![true; total_pages_in_group],
            }
        })
        .collect::<Vec<_>>();

    for predicate in predicates {
        // If the overall selection of a row group has ruled out all rows, there is
        // no need to evaluate the other predicates for it.
        let positions = (0..results.len())
            .filter(|&position| {
                results[position]
                    .overall_selection
                    .as_ref()
                    .is_none_or(|selection| selection.selects_any())
            })
            .collect::<Vec<_>>();
        if positions.is_empty() {
            break;
        }
        let predicate_row_groups = positions
            .iter()
            .map(|&position| row_groups[position])
            .collect::<Vec<_>>();
        let selections = prune_pages_for_predicate_in_row_groups(
            &predicate_row_groups,
            predicate,
            arrow_schema,
            parquet_schema,
            parquet_metadata,
            metrics,
        );

        for (position, selection) in positions.into_iter().zip(selections) {
            let Some((selection, pages)) = selection else {
                trace!("No pages pruned in prune_pages_in_one_row_group");
                continue;
            };

            debug!(
                "Use filter and page index to create RowSelection {:?} from predicate: {:?}",
                selection,
                predicate.predicate_expr(),
            );

            let result = &mut results[position];
            for (page, matched) in result.matched_pages.iter_mut().enumerate() {
                *matched &= pages.get(page).copied().unwrap_or(false);
            }
            result.overall_selection =
                update_selection(result.overall_selection.take(), selection);
        }
    }
    results
}

/// Like [`prune_pages_for_predicate`] for several row groups. The predicate is
/// evaluated once for the pages of all row groups. Falls back to pruning each row
/// group on its own if the page index of some row group is incomplete.
fn prune_pages_for_predicate_in_row_groups(
    row_groups: &[usize],
    predicate: &PruningPredicate,
    arrow_schema: &Schema,
    parquet_schema: &SchemaDescriptor,
    parquet_metadata: &ParquetMetaData,
    metrics: &ParquetFileMetrics,
) -> Vec<Option<(RowSelection, Vec<bool>)>> {
    #[cfg(test)]
    tests::PAGE_PREDICATE_CALLS.set(tests::PAGE_PREDICATE_CALLS.get() + row_groups.len());

    let Some(column) = predicate.required_columns().single_column() else {
        return vec![None; row_groups.len()];
    };
    let create_converter = || match StatisticsConverter::try_new(
        column.name(),
        arrow_schema,
        parquet_schema,
    ) {
        Ok(converter) => Some(converter),
        Err(e) => {
            debug!(
                "Could not create statistics converter for column {}: {e}",
                column.name()
            );
            None
        }
    };
    let Some(converter) = create_converter() else {
        return vec![None; row_groups.len()];
    };

    let Some(pruning_stats) = MultiRowGroupPagesPruningStatistics::try_new(
        row_groups,
        converter,
        parquet_metadata,
    ) else {
        return row_groups
            .iter()
            .map(|&row_group_index| {
                prune_pages_in_one_row_group(
                    row_group_index,
                    predicate,
                    create_converter()?,
                    parquet_metadata,
                    metrics,
                )
            })
            .collect();
    };

    let values = match predicate.prune(&pruning_stats) {
        Ok(values) => values,
        Err(e) => {
            debug!("Error evaluating page index predicate values {e}");
            metrics.predicate_evaluation_errors.add(row_groups.len());
            return vec![None; row_groups.len()];
        }
    };

    let mut offset = 0;
    pruning_stats
        .row_groups
        .iter()
        .map(|row_group| {
            let num_pages = row_group.page_offsets.len();
            let values = &values[offset..offset + num_pages];
            offset += num_pages;
            let Some(page_row_counts) =
                page_row_counts(row_group.num_rows, row_group.page_offsets)
            else {
                debug!(
                    "Can not determine page row counts for row group {}, skipping",
                    row_group.row_group_index
                );
                metrics.predicate_evaluation_errors.add(1);
                return None;
            };
            Some((row_selection(values, &page_row_counts), values.to_vec()))
        })
        .collect()
}

fn prune_pages_for_predicate(
    row_group_index: usize,
    predicate: &PruningPredicate,
    arrow_schema: &Schema,
    parquet_schema: &SchemaDescriptor,
    parquet_metadata: &ParquetMetaData,
    metrics: &ParquetFileMetrics,
) -> Option<(RowSelection, Vec<bool>)> {
    #[cfg(test)]
    tests::PAGE_PREDICATE_CALLS.set(tests::PAGE_PREDICATE_CALLS.get() + 1);

    let column = predicate.required_columns().single_column()?;
    let converter =
        match StatisticsConverter::try_new(column.name(), arrow_schema, parquet_schema) {
            Ok(converter) => converter,
            Err(e) => {
                debug!(
                    "Could not create statistics converter for column {}: {e}",
                    column.name()
                );
                return None;
            }
        };

    prune_pages_in_one_row_group(
        row_group_index,
        predicate,
        converter,
        parquet_metadata,
        metrics,
    )
}

/// Returns a [`RowSelection`] for the rows in this row group to scan, in addition to a vec of
/// booleans that state if each page was matched (true) or not (false).
///
/// This Row Selection is formed from the page index and the predicate skips row
/// ranges that can be ruled out based on the predicate.
///
/// Returns `None` if there is an error evaluating the predicate or the required
/// page information is not present.
fn prune_pages_in_one_row_group(
    row_group_index: usize,
    pruning_predicate: &PruningPredicate,
    converter: StatisticsConverter<'_>,
    parquet_metadata: &ParquetMetaData,
    metrics: &ParquetFileMetrics,
) -> Option<(RowSelection, Vec<bool>)> {
    let pruning_stats =
        PagesPruningStatistics::try_new(row_group_index, converter, parquet_metadata)?;

    // Each element in values is a boolean indicating whether the page may have
    // values that match the predicate (true) or could not possibly have values
    // that match the predicate (false).
    let values = match pruning_predicate.prune(&pruning_stats) {
        Ok(values) => values,
        Err(e) => {
            debug!("Error evaluating page index predicate values {e}");
            metrics.predicate_evaluation_errors.add(1);
            return None;
        }
    };

    // Convert the information of which pages to skip into a RowSelection
    // that describes the ranges of rows to skip.
    let Some(page_row_counts) = pruning_stats.page_row_counts() else {
        debug!(
            "Can not determine page row counts for row group {row_group_index}, skipping"
        );
        metrics.predicate_evaluation_errors.add(1);
        return None;
    };

    trace!("Pruned to {values:?} using {pruning_stats:?}");
    Some((row_selection(&values, &page_row_counts), values))
}

/// Creates a [`RowSelection`] from whether each page may match (`values`) and the
/// number of rows of each page.
fn row_selection(values: &[bool], page_row_counts: &[usize]) -> RowSelection {
    assert_eq!(page_row_counts.len(), values.len());
    let mut vec = Vec::with_capacity(values.len());
    let mut sum_row = page_row_counts[0];
    let mut selected = values[0];
    for (i, &f) in values.iter().enumerate().skip(1) {
        if f == selected {
            sum_row += page_row_counts[i];
        } else {
            let selector = if selected {
                RowSelector::select(sum_row)
            } else {
                RowSelector::skip(sum_row)
            };
            vec.push(selector);
            sum_row = page_row_counts[i];
            selected = f;
        }
    }

    let selector = if selected {
        RowSelector::select(sum_row)
    } else {
        RowSelector::skip(sum_row)
    };
    vec.push(selector);
    RowSelection::from(vec)
}

/// Returns the number of rows of each page of a row group with `num_rows` rows.
fn page_row_counts(num_rows: usize, page_offsets: &[PageLocation]) -> Option<Vec<usize>> {
    let mut vec = Vec::with_capacity(page_offsets.len());
    page_offsets.windows(2).for_each(|x| {
        let start = x[0].first_row_index as usize;
        let end = x[1].first_row_index as usize;
        vec.push(end - start);
    });
    vec.push(num_rows - page_offsets.last()?.first_row_index as usize);
    Some(vec)
}

/// Implement [`PruningStatistics`] for one column's [`PageIndexProvider`]
#[derive(Debug)]
struct PagesPruningStatistics<'a> {
    row_group_index: usize,
    row_group_metadatas: &'a [RowGroupMetaData],
    converter: StatisticsConverter<'a>,
    page_index: &'a dyn PageIndexProvider,
    page_offsets: &'a Vec<PageLocation>,
    trusted_min_max: bool,
}

impl<'a> PagesPruningStatistics<'a> {
    /// Creates a new [`PagesPruningStatistics`] for a column in a row group, if
    /// possible.
    ///
    /// Returns None if the `parquet_metadata` does not have sufficient
    /// information to create the statistics.
    fn try_new(
        row_group_index: usize,
        converter: StatisticsConverter<'a>,
        parquet_metadata: &'a ParquetMetaData,
    ) -> Option<Self> {
        let Some(parquet_column_index) = converter.parquet_column_index() else {
            trace!(
                "Column {:?} not in parquet file, skipping",
                converter.arrow_field()
            );
            return None;
        };

        let page_index: &dyn PageIndexProvider = parquet_metadata.page_index()?.as_ref();
        if !page_index.is_complete() {
            return None;
        }
        let row_group_metadatas = parquet_metadata.row_groups();

        let Some(offset_index_metadata) =
            page_index.offset_index(row_group_index, parquet_column_index)
        else {
            trace!(
                "No page offsets for column {:?} in row group {row_group_index}, skipping",
                converter.arrow_field()
            );
            return None;
        };
        let page_offsets = offset_index_metadata.page_locations();
        let file_metadata = parquet_metadata.file_metadata();
        let trusted_min_max = !has_untrusted_min_max_order(
            file_metadata.schema_descr(),
            file_metadata.column_orders().map(Vec::as_slice),
            parquet_column_index,
        );

        Some(Self {
            row_group_index,
            row_group_metadatas,
            converter,
            page_index,
            page_offsets,
            trusted_min_max,
        })
    }

    /// return the row counts in each data page, if possible.
    fn page_row_counts(&self) -> Option<Vec<usize>> {
        let row_group_metadata = self
            .row_group_metadatas
            .get(self.row_group_index)
            // fail fast/panic if row_group_index is out of bounds
            .unwrap();

        page_row_counts(row_group_metadata.num_rows() as usize, self.page_offsets)
    }
}
impl PruningStatistics for PagesPruningStatistics<'_> {
    fn min_values(&self, _column: &datafusion_common::Column) -> Option<ArrayRef> {
        if !self.trusted_min_max {
            return None;
        }
        match self
            .converter
            .data_page_mins(self.page_index, [&self.row_group_index])
        {
            Ok(min_values) => Some(min_values),
            Err(e) => {
                debug!("Error evaluating data page min values {e}");
                None
            }
        }
    }

    fn max_values(&self, _column: &datafusion_common::Column) -> Option<ArrayRef> {
        if !self.trusted_min_max {
            return None;
        }
        match self
            .converter
            .data_page_maxes(self.page_index, [&self.row_group_index])
        {
            Ok(min_values) => Some(min_values),
            Err(e) => {
                debug!("Error evaluating data page max values {e}");
                None
            }
        }
    }

    fn num_containers(&self) -> usize {
        self.page_offsets.len()
    }

    fn null_counts(&self, _column: &datafusion_common::Column) -> Option<ArrayRef> {
        match self
            .converter
            .data_page_null_counts(self.page_index, [&self.row_group_index])
        {
            Ok(null_counts) => Some(Arc::new(null_counts)),
            Err(e) => {
                debug!("Error evaluating data page null counts {e}");
                None
            }
        }
    }

    fn row_counts(&self) -> Option<ArrayRef> {
        match self.converter.data_page_row_counts(
            self.page_index,
            self.row_group_metadatas,
            [&self.row_group_index],
        ) {
            Ok(row_counts) => row_counts.map(|a| Arc::new(a) as ArrayRef),
            Err(e) => {
                debug!("Error evaluating data page row counts {e}");
                None
            }
        }
    }

    fn contained(
        &self,
        _column: &datafusion_common::Column,
        _values: &HashSet<ScalarValue>,
    ) -> Option<BooleanArray> {
        None
    }
}

/// The pages of one row group in [`MultiRowGroupPagesPruningStatistics`].
#[derive(Debug)]
struct RowGroupPageOffsets<'a> {
    row_group_index: usize,
    num_rows: usize,
    page_offsets: &'a [PageLocation],
}

/// Implements [`PruningStatistics`] for one column's [`PageIndexProvider`] in
/// several row groups. Each page is one container; the pages are ordered by row
/// group.
#[derive(Debug)]
struct MultiRowGroupPagesPruningStatistics<'a> {
    row_group_indexes: &'a [usize],
    row_groups: Vec<RowGroupPageOffsets<'a>>,
    row_group_metadatas: &'a [RowGroupMetaData],
    converter: StatisticsConverter<'a>,
    page_index: &'a dyn PageIndexProvider,
    num_pages: usize,
    trusted_min_max: bool,
}

impl<'a> MultiRowGroupPagesPruningStatistics<'a> {
    /// Creates a new [`MultiRowGroupPagesPruningStatistics`] for a column in the
    /// given row groups.
    ///
    /// Returns None if the `parquet_metadata` does not have sufficient
    /// information for all row groups.
    fn try_new(
        row_group_indexes: &'a [usize],
        converter: StatisticsConverter<'a>,
        parquet_metadata: &'a ParquetMetaData,
    ) -> Option<Self> {
        let parquet_column_index = converter.parquet_column_index()?;
        let page_index: &dyn PageIndexProvider = parquet_metadata.page_index()?.as_ref();
        if !page_index.is_complete() {
            return None;
        }
        let row_group_metadatas = parquet_metadata.row_groups();

        let row_groups = row_group_indexes
            .iter()
            .map(|&row_group_index| {
                let page_offsets = page_index
                    .offset_index(row_group_index, parquet_column_index)?
                    .page_locations();
                Some(RowGroupPageOffsets {
                    row_group_index,
                    num_rows: row_group_metadatas.get(row_group_index)?.num_rows()
                        as usize,
                    page_offsets,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        let num_pages = row_groups
            .iter()
            .map(|row_group| row_group.page_offsets.len())
            .sum();

        let file_metadata = parquet_metadata.file_metadata();
        let trusted_min_max = !has_untrusted_min_max_order(
            file_metadata.schema_descr(),
            file_metadata.column_orders().map(Vec::as_slice),
            parquet_column_index,
        );

        Some(Self {
            row_group_indexes,
            row_groups,
            row_group_metadatas,
            converter,
            page_index,
            num_pages,
            trusted_min_max,
        })
    }
}

impl PruningStatistics for MultiRowGroupPagesPruningStatistics<'_> {
    fn min_values(&self, _column: &datafusion_common::Column) -> Option<ArrayRef> {
        if !self.trusted_min_max {
            return None;
        }
        self.converter
            .data_page_mins(self.page_index, self.row_group_indexes)
            .inspect_err(|e| debug!("Error evaluating data page min values {e}"))
            .ok()
    }

    fn max_values(&self, _column: &datafusion_common::Column) -> Option<ArrayRef> {
        if !self.trusted_min_max {
            return None;
        }
        self.converter
            .data_page_maxes(self.page_index, self.row_group_indexes)
            .inspect_err(|e| debug!("Error evaluating data page max values {e}"))
            .ok()
    }

    fn num_containers(&self) -> usize {
        self.num_pages
    }

    fn null_counts(&self, _column: &datafusion_common::Column) -> Option<ArrayRef> {
        self.converter
            .data_page_null_counts(self.page_index, self.row_group_indexes)
            .inspect_err(|e| debug!("Error evaluating data page null counts {e}"))
            .ok()
            .map(|null_counts| Arc::new(null_counts) as ArrayRef)
    }

    fn row_counts(&self) -> Option<ArrayRef> {
        self.converter
            .data_page_row_counts(
                self.page_index,
                self.row_group_metadatas,
                self.row_group_indexes,
            )
            .inspect_err(|e| debug!("Error evaluating data page row counts {e}"))
            .ok()
            .flatten()
            .map(|row_counts| Arc::new(row_counts) as ArrayRef)
    }

    fn contained(
        &self,
        _column: &datafusion_common::Column,
        _values: &HashSet<ScalarValue>,
    ) -> Option<BooleanArray> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int32Array;
    use arrow::datatypes::{DataType, Field};
    use arrow::record_batch::RecordBatch;
    use bytes::Bytes;
    use datafusion_expr::Operator;
    use datafusion_physical_expr::expressions::{BinaryExpr, Column, lit};
    use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
    use parquet::arrow::ArrowWriter;
    use parquet::arrow::arrow_reader::{
        ArrowReaderOptions, ParquetRecordBatchReaderBuilder,
    };
    use parquet::basic::Type as PhysicalType;
    use parquet::file::metadata::{ColumnChunkMetaData, PageIndexPolicy};
    use parquet::file::properties::WriterProperties;
    use parquet::schema::types::{SchemaDescriptor, Type as SchemaType};

    thread_local! {
        pub(super) static PAGE_PREDICATE_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    #[test]
    fn inverse_pruning_short_circuits() {
        // Each row group has three two-row pages. Only the middle page can
        // prove a > 5; mixed pages must still be scanned by ordinary pruning.
        for (middle, limit, selected, fully_matched, expected_calls, expected_rows) in [
            ([0, 10], Some(2), 6, false, 9, 18), // Empty inverse: skip the second conjunct.
            ([10, 10], Some(2), 6, false, 8, 2), // First group covers LIMIT.
            ([10, 10], Some(3), 6, false, 10, 4), // Two groups cover LIMIT.
            ([10, 10], Some(7), 6, false, 12, 18), // Insufficient proof: retain the plan.
            ([10, 10], None, 6, false, 6, 18),   // Ordinary pruning only.
            ([10, 10], Some(0), 6, false, 6, 0),
            ([10, 10], Some(2), 3, false, 10, 3), // Only one guaranteed row in RG0.
            ([10, 10], Some(2), 1, true, 6, 3), // Fully matched RG0 contributes one row.
        ] {
            let schema =
                Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
            let props = WriterProperties::builder()
                .set_data_page_row_count_limit(2)
                .set_write_batch_size(2)
                .build();
            let mut writer =
                ArrowWriter::try_new(Vec::new(), Arc::clone(&schema), Some(props))
                    .unwrap();
            for row_group in 0..3 {
                let values = if fully_matched && row_group == 0 {
                    vec![10; 6]
                } else {
                    vec![0, 10, middle[0], middle[1], 0, 10]
                };
                let batch = RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![Arc::new(Int32Array::from(values))],
                )
                .unwrap();
                writer.write(&batch).unwrap();
                writer.flush().unwrap();
            }
            let options = ArrowReaderOptions::new()
                .with_page_index_policy(PageIndexPolicy::Required);
            let reader = ParquetRecordBatchReaderBuilder::try_new_with_options(
                Bytes::from(writer.into_inner().unwrap()),
                options,
            )
            .unwrap();
            let column: Arc<dyn PhysicalExpr> = Arc::new(Column::new("a", 0));
            let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
                Arc::new(BinaryExpr::new(
                    Arc::clone(&column),
                    Operator::Gt,
                    lit(5i32),
                )),
                Operator::And,
                Arc::new(BinaryExpr::new(column, Operator::Lt, lit(20i32))),
            ));
            let filter = PagePruningAccessPlanFilter::new(&expr, Arc::clone(&schema));
            let mut plan = ParquetAccessPlan::new_all(3);
            plan.scan_selection(
                0,
                vec![
                    RowSelector::select(selected),
                    RowSelector::skip(6 - selected),
                ]
                .into(),
            );
            if fully_matched {
                plan.mark_fully_matched(0);
            }
            let metrics =
                ParquetFileMetrics::new(0, "test", &ExecutionPlanMetricsSet::new());
            PAGE_PREDICATE_CALLS.set(0);
            let result = filter.prune_plan_with_page_index_and_metrics(
                plan,
                &schema,
                reader.parquet_schema(),
                reader.metadata(),
                &metrics,
                limit,
            );
            assert_eq!(
                PAGE_PREDICATE_CALLS.get(),
                expected_calls,
                "middle={middle:?}, limit={limit:?}, selected={selected}, fully_matched={fully_matched}"
            );
            assert_eq!(
                selected_row_count(&result.access_plan, reader.metadata().row_groups()),
                expected_rows
            );
        }
    }

    #[test]
    fn limit_plan_preserves_existing_selections() {
        let groups = vec![row_group_metadata(6), row_group_metadata(6)];
        let mut access_plan = ParquetAccessPlan::new(vec![
            RowGroupAccess::Selection(
                vec![
                    RowSelector::skip(2),
                    RowSelector::select(1),
                    RowSelector::skip(3),
                ]
                .into(),
            ),
            RowGroupAccess::Selection(
                vec![
                    RowSelector::select(2),
                    RowSelector::skip(1),
                    RowSelector::select(3),
                ]
                .into(),
            ),
        ]);
        access_plan.mark_fully_matched(0);

        let fully_matched_pages = vec![
            None,
            Some(
                vec![
                    RowSelector::skip(1),
                    RowSelector::select(4),
                    RowSelector::skip(1),
                ]
                .into(),
            ),
        ];

        // One selected row from the fully matched row group plus three rows
        // from the fully matched page satisfy the limit.
        let (plan, pruned_rows, _) =
            limit_pruned_plan(&access_plan, &fully_matched_pages, 4, &groups).unwrap();
        assert_eq!(selected_row_count(&plan, &groups), 4);
        assert_eq!(pruned_rows, 2);
        assert!(plan.is_fully_matched(0));
        assert_eq!(plan.inner()[0], access_plan.inner()[0]);
        assert_eq!(
            plan.inner()[1],
            RowGroupAccess::Selection(
                vec![
                    RowSelector::skip(1),
                    RowSelector::select(1),
                    RowSelector::skip(1),
                    RowSelector::select(2),
                    RowSelector::skip(1),
                ]
                .into()
            )
        );

        // The same candidates are insufficient for a larger limit.
        assert!(
            limit_pruned_plan(&access_plan, &fully_matched_pages, 5, &groups).is_none()
        );
    }

    fn row_group_metadata(num_rows: i64) -> RowGroupMetaData {
        let field = SchemaType::primitive_type_builder("a", PhysicalType::INT32)
            .build()
            .unwrap();
        let schema = Arc::new(SchemaDescriptor::new(Arc::new(
            SchemaType::group_type_builder("schema")
                .with_fields(vec![Arc::new(field)])
                .build()
                .unwrap(),
        )));
        let column = ColumnChunkMetaData::builder(schema.column(0))
            .set_num_values(num_rows)
            .build()
            .unwrap();

        RowGroupMetaData::builder(schema)
            .set_num_rows(num_rows)
            .set_column_metadata(vec![column])
            .build()
            .unwrap()
    }
}
