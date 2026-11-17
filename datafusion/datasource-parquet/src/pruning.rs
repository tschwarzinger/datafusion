// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Shared predicate construction for row-group and page-index pruning.

use std::collections::VecDeque;
use std::sync::Arc;

use crate::page_filter::PagePruningAccessPlanFilter;
use arrow::datatypes::{Schema, SchemaRef};
use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion_expr::Operator;
use datafusion_physical_expr::expressions::{BinaryExpr, IsNullExpr, NotExpr};
use datafusion_physical_expr::utils::collect_columns;
use datafusion_physical_expr::{PhysicalExpr, PhysicalExprSimplifier};
use datafusion_physical_plan::metrics::Count;
use datafusion_pruning::{PruningPredicate, PruningPredicateBuilder};
use parking_lot::Mutex;

/// Build a null-safe inverse used to prove every row matches `predicate`.
///
/// Rows where a filter evaluates to NULL do not pass it, so nullable referenced
/// columns are included in the inverse. If the inverse can be pruned, every row
/// is guaranteed to satisfy the original predicate.
pub(crate) fn build_inverted_predicate(
    predicate: &PruningPredicate,
    arrow_schema: &Schema,
) -> Option<PruningPredicate> {
    // Some pruning rewrites preserve only whether rows can be TRUE, while
    // full-match inference must distinguish FALSE from UNKNOWN.
    if !predicate.can_be_inverted_for_full_match() {
        return None;
    }
    let mut inverted_expr: Arc<dyn PhysicalExpr> =
        Arc::new(NotExpr::new(Arc::clone(predicate.orig_expr())));

    let mut columns = collect_columns(predicate.orig_expr())
        .into_iter()
        .filter(|column| arrow_schema.field(column.index()).is_nullable())
        .collect::<Vec<_>>();
    columns.sort_by(|a, b| {
        a.index()
            .cmp(&b.index())
            .then_with(|| a.name().cmp(b.name()))
    });

    for column in columns {
        inverted_expr = Arc::new(BinaryExpr::new(
            inverted_expr,
            Operator::Or,
            Arc::new(IsNullExpr::new(Arc::new(column))),
        ));
    }

    let simplifier = PhysicalExprSimplifier::new(arrow_schema);
    let inverted_expr = simplifier.simplify(inverted_expr).ok()?;
    PruningPredicateBuilder::new()
        .with_file_schema(Arc::clone(predicate.schema()))
        .with_max_in_list_size(predicate.max_in_list_size())
        .try_build(inverted_expr)
        .ok()
}

/// Pruning predicates that files with the same schema and predicate share.
#[derive(Debug)]
pub(crate) struct FilePruningPredicates {
    /// Prunes row groups by statistics and Bloom filters.
    pub(crate) row_groups: Option<Arc<PruningPredicate>>,
    /// Prunes pages by the page index, if enabled.
    pub(crate) pages: Option<Arc<PagePruningAccessPlanFilter>>,
}

/// Shares the pruning predicates that a scan builds for its files.
///
/// Building pruning predicates is costly for large predicates, such as the
/// `IN` lists of dynamic join filters, while the files of a scan usually have
/// the same physical schema and thus the same predicate once it is adapted to
/// the file. The cache returns the predicates built for an earlier file if the
/// file schema and the adapted predicate are equal and its dynamic filters
/// have not been updated since.
#[derive(Debug)]
pub(crate) struct PruningPredicateCache {
    enable_page_index: bool,
    max_in_list_size: usize,
    /// The most recently built entries, the newest last.
    entries: Mutex<VecDeque<Arc<PruningPredicateCacheEntry>>>,
}

#[derive(Debug)]
struct PruningPredicateCacheEntry {
    file_schema: SchemaRef,
    predicate: Arc<dyn PhysicalExpr>,
    generations: Vec<u64>,
    predicates: Arc<FilePruningPredicates>,
    /// Predicate creation errors while building `predicates`, which are
    /// counted again on every use, as if the predicates had been rebuilt.
    creation_errors: usize,
}

impl PruningPredicateCache {
    /// The number of entries kept. A scan rarely needs more than one, unless
    /// its files have different schemas.
    const CAPACITY: usize = 4;

    pub(crate) fn new(enable_page_index: bool, max_in_list_size: usize) -> Self {
        Self {
            enable_page_index,
            max_in_list_size,
            entries: Mutex::new(VecDeque::with_capacity(Self::CAPACITY)),
        }
    }

    /// Returns the pruning predicates of `predicate` for a file with
    /// `file_schema`, building them if no earlier file had equal ones.
    pub(crate) fn get_or_build(
        &self,
        predicate: &Arc<dyn PhysicalExpr>,
        file_schema: &SchemaRef,
        predicate_creation_errors: &Count,
    ) -> Arc<FilePruningPredicates> {
        let generations = dynamic_filter_generations(predicate);
        let cached = self
            .entries
            .lock()
            .iter()
            .rev()
            .find(|entry| {
                entry.generations == generations
                    && (Arc::ptr_eq(&entry.file_schema, file_schema)
                        || entry.file_schema == *file_schema)
                    && (Arc::ptr_eq(&entry.predicate, predicate)
                        || entry.predicate.as_ref() == predicate.as_ref())
            })
            .cloned();
        if let Some(entry) = cached {
            predicate_creation_errors.add(entry.creation_errors);
            return Arc::clone(&entry.predicates);
        }

        // A dynamic filter may be updated while the predicates are built. They
        // are then built from a newer snapshot than `generations` names, which
        // is still a valid filter for any file.
        let creation_errors = Count::new();
        let row_groups = PruningPredicateBuilder::new()
            .with_file_schema(Arc::clone(file_schema))
            .with_error_counter(&creation_errors)
            .with_max_in_list_size(self.max_in_list_size)
            .build(Arc::clone(predicate));
        let pages = self
            .enable_page_index
            .then(|| {
                PagePruningAccessPlanFilter::new_with_max_in_list_size(
                    predicate,
                    file_schema,
                    self.max_in_list_size,
                )
            })
            .filter(|pages| pages.filter_number() > 0)
            .map(Arc::new);
        let predicates = Arc::new(FilePruningPredicates { row_groups, pages });
        predicate_creation_errors.add(creation_errors.value());

        let mut entries = self.entries.lock();
        if entries.len() == Self::CAPACITY {
            entries.pop_front();
        }
        entries.push_back(Arc::new(PruningPredicateCacheEntry {
            file_schema: Arc::clone(file_schema),
            predicate: Arc::clone(predicate),
            generations,
            predicates: Arc::clone(&predicates),
            creation_errors: creation_errors.value(),
        }));
        predicates
    }
}

/// Returns the generations of the dynamic filters in `predicate`, in the order
/// of a pre-order traversal. Two equal predicates contain the same dynamic
/// filters in the same order, as [`DynamicFilterPhysicalExpr`]s are only
/// equal if they share their source.
///
/// [`DynamicFilterPhysicalExpr`]: datafusion_physical_expr::expressions::DynamicFilterPhysicalExpr
fn dynamic_filter_generations(predicate: &Arc<dyn PhysicalExpr>) -> Vec<u64> {
    let mut generations = vec![];
    predicate
        .apply(|expr| {
            let generation = expr.snapshot_generation();
            if generation != 0 {
                generations.push(generation);
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .expect("traversal is infallible");
    generations
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::datatypes::{DataType, Field};
    use datafusion_expr::{col, lit};
    use datafusion_physical_expr::expressions::{Column, DynamicFilterPhysicalExpr};
    use datafusion_physical_expr::planner::logical2physical;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]))
    }

    #[test]
    fn pruning_predicate_cache_shares_equal_predicates() {
        let cache = PruningPredicateCache::new(true, 20);
        let errors = Count::new();
        let expr = col("a")
            .in_list(vec![lit(1), lit(2)], false)
            .and(col("b").gt(lit(5)));

        // Equal, but separately built schemas and predicates share the entry.
        let first =
            cache.get_or_build(&logical2physical(&expr, &schema()), &schema(), &errors);
        let second =
            cache.get_or_build(&logical2physical(&expr, &schema()), &schema(), &errors);
        assert!(Arc::ptr_eq(&first, &second));
        assert!(first.row_groups.is_some());
        assert_eq!(first.pages.as_ref().unwrap().filter_number(), 2);

        // A different predicate or schema builds new predicates.
        let other = logical2physical(&col("a").eq(lit(1)), &schema());
        assert!(!Arc::ptr_eq(
            &first,
            &cache.get_or_build(&other, &schema(), &errors)
        ));
        let other_schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int32, false),
        ]));
        let predicate = logical2physical(&expr, &other_schema);
        assert!(!Arc::ptr_eq(
            &first,
            &cache.get_or_build(&predicate, &other_schema, &errors)
        ));
        assert_eq!(errors.value(), 0);
    }

    #[test]
    fn pruning_predicate_cache_rebuilds_on_dynamic_filter_update() {
        let cache = PruningPredicateCache::new(false, 20);
        let errors = Count::new();
        let column = Arc::new(Column::new("a", 0)) as Arc<dyn PhysicalExpr>;
        let dynamic = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![Arc::clone(&column)],
            logical2physical(&lit(true), &schema()),
        ));
        let predicate = Arc::clone(&dynamic) as Arc<dyn PhysicalExpr>;

        let before = cache.get_or_build(&predicate, &schema(), &errors);
        assert!(Arc::ptr_eq(
            &before,
            &cache.get_or_build(&predicate, &schema(), &errors)
        ));
        assert!(before.row_groups.is_none());
        assert!(before.pages.is_none());

        dynamic
            .update(logical2physical(&col("a").eq(lit(1)), &schema()))
            .unwrap();
        let after = cache.get_or_build(&predicate, &schema(), &errors);
        assert!(!Arc::ptr_eq(&before, &after));
        let row_groups = after.row_groups.as_ref().unwrap();
        assert_eq!(row_groups.literal_guarantees().len(), 1);
    }
}
