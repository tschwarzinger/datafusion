// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! A cheap, lazily-computed summary of a [`LogicalPlan`] that lets optimizer
//! rules quickly *rule themselves out* without walking or cloning the plan.
//!
//! [`LogicalPlanSketch`] records, in a single traversal, which plan-node
//! variants, expression variants, join types, and extension nodes are present.
//! It is a *conservative* filter: if the sketch says a rule "may change" the
//! plan, the rule is always run; it may only be skipped when the sketch proves
//! (by absence of the relevant node/expr/join types) that the rule cannot
//! possibly rewrite the plan.
//!
//! [`LazyLogicalPlanSketch`] defers computing the full sketch until a rule
//! actually asks for it, and caches the result so that, while the plan is
//! unchanged (i.e. between rules that return `Transformed::no`), the sketch is
//! computed at most once per plan version. The optimizer invalidates the cache
//! whenever a rule changes the plan (returns `Transformed::yes`).

use std::collections::HashSet;

use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion_expr::{Expr, JoinType, LogicalPlan};

/// A classified subset of [`LogicalPlan`] variants used for cheap gating.
///
/// This is intentionally a curated list of the node kinds optimizer rules care
/// about; it is not (currently) a bijection onto `LogicalPlan`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeKind {
    Projection,
    Filter,
    Aggregate,
    Window,
    Sort,
    Join,
    Limit,
    Union,
    Distinct,
    EmptyRelation,
    Repartition,
    Unnest,
    Subquery,
}

/// A classified subset of [`Expr`] variants used for cheap gating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExprKind {
    Exists,
    InSubquery,
    ScalarSubquery,
    SetComparison,
    Literal,
    Column,
    Alias,
    BinaryExpr,
    Not,
    IsNull,
    IsNotNull,
    Case,
    Cast,
    TryCast,
    InList,
    Like,
    AggregateFunction,
    ScalarFunction,
    WindowFunction,
}

impl NodeKind {
    fn bit(self) -> u128 {
        1u128 << (self as u32)
    }
}

impl ExprKind {
    fn bit(self) -> u128 {
        1u128 << (self as u32)
    }
}

fn node_kind(node: &LogicalPlan) -> Option<NodeKind> {
    match node {
        LogicalPlan::Projection(_) => Some(NodeKind::Projection),
        LogicalPlan::Filter(_) => Some(NodeKind::Filter),
        LogicalPlan::Aggregate(_) => Some(NodeKind::Aggregate),
        LogicalPlan::Window(_) => Some(NodeKind::Window),
        LogicalPlan::Sort(_) => Some(NodeKind::Sort),
        LogicalPlan::Join(_) => Some(NodeKind::Join),
        LogicalPlan::Limit(_) => Some(NodeKind::Limit),
        LogicalPlan::Union(_) => Some(NodeKind::Union),
        LogicalPlan::Distinct(_) => Some(NodeKind::Distinct),
        LogicalPlan::EmptyRelation(_) => Some(NodeKind::EmptyRelation),
        LogicalPlan::Repartition(_) => Some(NodeKind::Repartition),
        LogicalPlan::Unnest(_) => Some(NodeKind::Unnest),
        LogicalPlan::Subquery(_) => Some(NodeKind::Subquery),
        // Tables, DDL/DML, statements, explains, values, extension, etc. contribute no gate bit.
        _ => None,
    }
}

fn expr_kind(expr: &Expr) -> Option<ExprKind> {
    match expr {
        Expr::Exists(_) => Some(ExprKind::Exists),
        Expr::InSubquery(_) => Some(ExprKind::InSubquery),
        Expr::ScalarSubquery(_) => Some(ExprKind::ScalarSubquery),
        Expr::SetComparison(_) => Some(ExprKind::SetComparison),
        Expr::Literal(_, _) => Some(ExprKind::Literal),
        Expr::Column(_) => Some(ExprKind::Column),
        Expr::Alias(_) => Some(ExprKind::Alias),
        Expr::BinaryExpr(_) => Some(ExprKind::BinaryExpr),
        Expr::Not(_) => Some(ExprKind::Not),
        Expr::IsNull(_) => Some(ExprKind::IsNull),
        Expr::IsNotNull(_) => Some(ExprKind::IsNotNull),
        Expr::Case(_) => Some(ExprKind::Case),
        Expr::Cast(_) => Some(ExprKind::Cast),
        Expr::TryCast(_) => Some(ExprKind::TryCast),
        Expr::InList(_) => Some(ExprKind::InList),
        Expr::Like(_) | Expr::SimilarTo(_) => Some(ExprKind::Like),
        Expr::AggregateFunction(_) => Some(ExprKind::AggregateFunction),
        Expr::ScalarFunction(_) => Some(ExprKind::ScalarFunction),
        Expr::WindowFunction(_) => Some(ExprKind::WindowFunction),
        _ => None,
    }
}

fn join_bit(join_type: JoinType) -> u16 {
    // 10 variants: Inner, Left, Right, Full, LeftSemi, RightSemi, LeftAnti,
    // RightAnti, LeftMark, RightMark -> use a contiguous u16 bitmask.
    let idx = match join_type {
        JoinType::Inner => 0,
        JoinType::Left => 1,
        JoinType::Right => 2,
        JoinType::Full => 3,
        JoinType::LeftSemi => 4,
        JoinType::RightSemi => 5,
        JoinType::LeftAnti => 6,
        JoinType::RightAnti => 7,
        JoinType::LeftMark => 8,
        JoinType::RightMark => 9,
    };
    1u16 << idx
}

/// A cheap summary of a [`LogicalPlan`] (see the module docs).
#[derive(Debug, Clone, Default)]
pub struct LogicalPlanSketch {
    nodes: u128,
    exprs: u128,
    joins: u16,
    /// `true` if any subquery expression (`EXISTS` / `IN` / scalar / set) exists.
    has_subqueries: bool,
    /// Names of the distinct extension nodes present in the plan.
    extension_nodes: HashSet<String>,
}

impl LogicalPlanSketch {
    /// Compute the sketch for `plan` in a single traversal.
    pub fn new(plan: &LogicalPlan) -> Self {
        let mut sketch = Self::default();
        sketch.collect(plan);
        sketch
    }

    fn collect(&mut self, plan: &LogicalPlan) {
        plan.apply(|node| {
            if let Some(kind) = node_kind(node) {
                self.nodes |= kind.bit();
            }
            if let LogicalPlan::Join(join) = node {
                self.joins |= join_bit(join.join_type);
            }
            if let LogicalPlan::Extension(ext) = node {
                self.extension_nodes.insert(ext.node.name().to_string());
            }
            node.apply_expressions(|expr| {
                self.collect_expr(expr)?;
                Ok(TreeNodeRecursion::Continue)
            })?;
            Ok(TreeNodeRecursion::Continue)
        })
        .expect("sketch traversal should not error");
    }

    /// Record the expression variants in `expr` and, for subquery expressions,
    /// descend into the subquery plans they reference.
    fn collect_expr(&mut self, expr: &Expr) -> Result<(), datafusion_common::DataFusionError> {
        expr.apply(|e| {
            if let Some(kind) = expr_kind(e) {
                self.exprs |= kind.bit();
            }
            match e {
                Expr::Exists(ex) => {
                    self.has_subqueries = true;
                    self.collect(&ex.subquery.subquery);
                }
                Expr::ScalarSubquery(sq) => {
                    self.has_subqueries = true;
                    self.collect(&sq.subquery);
                }
                Expr::InSubquery(sq) => {
                    self.has_subqueries = true;
                    self.collect(&sq.subquery.subquery);
                }
                Expr::SetComparison(sc) => {
                    self.has_subqueries = true;
                    self.collect(&sc.subquery.subquery);
                }
                _ => {}
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        Ok(())
    }

    /// Whether this node (or any subquery) is present in the plan.
    pub fn has_node(&self, kind: NodeKind) -> bool {
        self.nodes & kind.bit() != 0
    }

    /// Whether this expression variant is present in the plan.
    pub fn has_expr(&self, kind: ExprKind) -> bool {
        self.exprs & kind.bit() != 0
    }

    /// Whether a join of exactly `join_type` is present.
    pub fn has_join_type(&self, join_type: JoinType) -> bool {
        self.joins & join_bit(join_type) != 0
    }

    /// Whether any join type in `join_types` is present.
    pub fn has_any_join_type(&self, join_types: &[JoinType]) -> bool {
        join_types
            .iter()
            .any(|t| self.joins & join_bit(*t) != 0)
    }

    /// Whether any subquery expression is present anywhere in the plan
    /// (including nested subquery plans).
    pub fn has_subqueries(&self) -> bool {
        self.has_subqueries
    }

    /// The names of the distinct extension nodes present in the plan.
    pub fn extension_nodes(&self) -> &HashSet<String> {
        &self.extension_nodes
    }
}

/// A lazily computed and cached [`LogicalPlanSketch`].
///
/// Computing the sketch is avoided entirely until a rule asks for it; once
/// computed, it is reused for as long as the plan is unchanged (that is, until
/// [`Self::invalidate`] is called). This is what allows a chain of `no`-op
/// rules to share a single sketch computation.
#[derive(Debug, Default)]
pub struct LazyLogicalPlanSketch {
    cached: Option<LogicalPlanSketch>,
}

impl LazyLogicalPlanSketch {
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the sketch for `plan`, computing it now if it has not been
    /// computed for the current plan version yet (or if it was invalidated).
    pub fn sketch(&mut self, plan: &LogicalPlan) -> &LogicalPlanSketch {
        if self.cached.is_none() {
            let start = datafusion_common::instant::Instant::now();
            self.cached = Some(LogicalPlanSketch::new(plan));
            eprintln!(
                "optimizer sketch: duration_us={}",
                start.elapsed().as_micros()
            );
        }
        self.cached.as_ref().expect("sketch was just computed")
    }

    /// Forget the cached sketch; call this whenever the plan changes so the
    /// next [`Self::sketch`] reflects the new plan.
    pub fn invalidate(&mut self) {
        self.cached = None;
    }
}
