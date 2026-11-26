// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Measures, for every TPC-H query, which logical and physical optimizer rule
//! invocations actually change the plan, and how much time is spent in
//! invocations that do not.
//!
//! Output: one JSON line per (query, phase, pass, rule) with median timings,
//! followed by an "oracle" line per query that re-runs the optimizers with only
//! the rules that changed something and reports the achieved time.

use std::collections::HashSet;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use datafusion_common::instant::Instant;

use clap::Parser;
use datafusion::common::ScalarValue;
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::datasource::MemTable;
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::context::SessionState;
use datafusion::logical_expr::{
    Expr, ExprSchemable, ExpressionPlacement, JoinType, LogicalPlan,
};
use datafusion::optimizer::{Optimizer, OptimizerRule};
use datafusion::physical_optimizer::{PhysicalOptimizerContext, PhysicalOptimizerRule};
use datafusion::physical_plan::{ExecutionPlan, ExecutionPlanProperties, displayable};
use datafusion::prelude::*;
use datafusion_benchmarks::tpch::{TPCH_TABLES, get_tpch_table_schema};
use datafusion_common::Result;
use serde_json::json;

#[derive(Parser)]
struct Args {
    /// Directory with TPC-H parquet files (`<table>.parquet` or `<table>/`).
    /// If omitted, empty in-memory tables are used.
    #[arg(long)]
    data: Option<String>,
    /// Directory containing q1.sql .. q22.sql
    #[arg(long, default_value = "benchmarks/queries")]
    queries: String,
    #[arg(long, default_value_t = 30)]
    iterations: usize,
    #[arg(long, default_value_t = 4)]
    partitions: usize,
    /// Only run physical planning of this query `--loops` times (for profilers)
    #[arg(long)]
    profile_query: Option<usize>,
    #[arg(long, default_value_t = 100)]
    loops: usize,
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn phys_string(p: &Arc<dyn ExecutionPlan>) -> String {
    displayable(p.as_ref())
        .set_show_schema(true)
        .indent(true)
        .to_string()
}

#[derive(Clone, Debug)]
struct PhysRecord {
    name: String,
    dur: Duration,
    ptr_same: bool,
    changed: bool,
    rebuild: RebuildStats,
}

/// How many nodes of a rule's output are fresh allocations, and how many of
/// those equal (same node description, same descendants) an input subtree.
#[derive(Clone, Debug, Default)]
struct RebuildStats {
    nodes: usize,
    new_nodes: usize,
    needless: usize,
    /// new nodes whose counterpart in the input (same description and schema)
    /// has identical plan properties
    same_props: usize,
    /// new nodes whose children all have properties identical to the input
    /// counterpart's children, and whose own properties are identical too
    same_props_and_child_props: usize,
    new_by_type: std::collections::BTreeMap<String, (usize, usize)>,
}

fn subtree_fps(
    p: &Arc<dyn ExecutionPlan>,
    out: &mut Vec<(Arc<dyn ExecutionPlan>, u64)>,
) -> u64 {
    let mut h = DefaultHasher::new();
    displayable(p.as_ref()).one_line().to_string().hash(&mut h);
    p.schema().hash(&mut h);
    for c in p.children() {
        subtree_fps(c, out).hash(&mut h);
    }
    let fp = h.finish();
    out.push((Arc::clone(p), fp));
    fp
}

fn rebuild_stats(
    before: &Arc<dyn ExecutionPlan>,
    after: &Arc<dyn ExecutionPlan>,
) -> RebuildStats {
    let (mut b, mut a) = (vec![], vec![]);
    subtree_fps(before, &mut b);
    subtree_fps(after, &mut a);
    let ptrs: HashSet<*const ()> =
        b.iter().map(|(p, _)| Arc::as_ptr(p).cast::<()>()).collect();
    let fps: HashSet<u64> = b.iter().map(|(_, f)| *f).collect();
    let key = |p: &Arc<dyn ExecutionPlan>| {
        let mut h = DefaultHasher::new();
        displayable(p.as_ref()).one_line().to_string().hash(&mut h);
        p.schema().hash(&mut h);
        h.finish()
    };
    let props = |p: &Arc<dyn ExecutionPlan>| {
        format!(
            "{} | {:?} | {:?} | {:?}",
            p.equivalence_properties(),
            p.output_partitioning(),
            p.pipeline_behavior(),
            p.boundedness()
        )
    };
    let mut in_props: std::collections::HashMap<u64, Vec<(String, Vec<String>)>> =
        Default::default();
    for (p, _) in &b {
        in_props
            .entry(key(p))
            .or_default()
            .push((props(p), p.children().iter().map(|c| props(c)).collect()));
    }
    let mut st = RebuildStats {
        nodes: a.len(),
        ..Default::default()
    };
    for (p, f) in &a {
        if !ptrs.contains(&(Arc::as_ptr(p).cast::<()>())) {
            st.new_nodes += 1;
            let needless = fps.contains(f);
            let mine = props(p);
            let kids: Vec<String> = p.children().iter().map(|c| props(c)).collect();
            if let Some(cands) = in_props.get(&key(p)) {
                if cands.iter().any(|(pp, _)| *pp == mine) {
                    st.same_props += 1;
                }
                if cands.iter().any(|(pp, ck)| *pp == mine && *ck == kids) {
                    st.same_props_and_child_props += 1;
                }
            }
            st.needless += needless as usize;
            let e = st.new_by_type.entry(p.name().to_string()).or_default();
            e.0 += 1;
            e.1 += needless as usize;
        }
    }
    st
}

#[derive(Debug)]
struct Recording {
    inner: Arc<dyn PhysicalOptimizerRule + Send + Sync>,
    log: Arc<Mutex<Vec<PhysRecord>>>,
}

impl PhysicalOptimizerRule for Recording {
    fn optimize(
        &self,
        _plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        unreachable!()
    }

    fn optimize_with_context(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        context: &dyn PhysicalOptimizerContext,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let before = Arc::clone(&plan);
        let start = Instant::now();
        let out = self.inner.optimize_with_context(plan, context)?;
        let dur = start.elapsed();
        let ptr_same = Arc::ptr_eq(&before, &out);
        let changed = !ptr_same && phys_string(&before) != phys_string(&out);
        let rebuild = rebuild_stats(&before, &out);
        self.log.lock().unwrap().push(PhysRecord {
            name: self.inner.name().to_string(),
            dur,
            ptr_same,
            changed,
            rebuild,
        });
        Ok(out)
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn schema_check(&self) -> bool {
        self.inner.schema_check()
    }
}

#[derive(Clone)]
struct LogRecord {
    pass: usize,
    name: String,
    dur: Duration,
    changed: bool,
    trigger: Option<bool>,
    trigger_dur: Duration,
    /// CSE: some node repeats a non-trivial subexpression (or has a Window)
    cse_dup: Option<bool>,
    /// simplify_expressions: (nodes not seen in a previous simplify output, total nodes)
    simp_dirty: Option<(usize, usize, usize, usize, usize)>,
    prop_dur: Duration,
}

/// `common_sub_expression_eliminate` only rewrites a node if one of its
/// non-ignored subexpressions occurs at least twice. Window nodes are CSEd
/// together with adjacent windows, so they are reported as "might fire".
fn cse_might_fire(plan: &LogicalPlan) -> bool {
    use LogicalPlan as P;
    any_node(plan, |n| match n {
        P::Window(_) => true,
        P::Projection(_) | P::Sort(_) | P::Filter(_) | P::Aggregate(_) => {
            let mut seen = HashSet::new();
            let mut dup = false;
            let _ = n.apply_expressions(|e| {
                e.apply(|x| {
                    let ignored = matches!(
                        x,
                        Expr::Literal(..)
                            | Expr::Column(..)
                            | Expr::ScalarVariable(..)
                            | Expr::Alias(..)
                            | Expr::Lambda(_)
                            | Expr::LambdaVariable(_)
                            | Expr::WindowFunction(..)
                    );
                    // a hash collision only errs towards "might fire"
                    let mut h = DefaultHasher::new();
                    x.hash(&mut h);
                    if !ignored && !seen.insert(h.finish()) {
                        dup = true;
                        return Ok(TreeNodeRecursion::Stop);
                    }
                    Ok(TreeNodeRecursion::Continue)
                })
            });
            dup
        }
        _ => false,
    })
}

fn sorted_refs(e: &Expr) -> Vec<&Column> {
    let mut v: Vec<_> = e.column_refs().into_iter().collect();
    v.sort();
    v
}

/// Per expression: the expression, whether it sits in a Join (simplified without
/// canonicalization) and the type/nullability of every column it references,
/// which is all of the schema the simplifier can observe.
fn simplify_expr_fingerprints(plan: &LogicalPlan) -> Vec<u64> {
    let mut out = vec![];
    let _ = plan.apply_with_subqueries(|n| {
        let inputs = n.inputs();
        let is_join = matches!(n, LogicalPlan::Join(_));
        let _ = n.apply_expressions(|e| {
            // nothing to simplify in a bare column/literal (possibly aliased)
            let trivial = !e
                .exists(|x| {
                    Ok(!matches!(
                        x,
                        Expr::Column(_) | Expr::Literal(..) | Expr::Alias(_)
                    ))
                })
                .unwrap();
            if trivial {
                return Ok(TreeNodeRecursion::Continue);
            }
            let mut h = DefaultHasher::new();
            is_join.hash(&mut h);
            e.hash(&mut h);
            for c in sorted_refs(e) {
                let field = inputs.iter().find_map(|i| {
                    i.schema()
                        .qualified_field_from_column(c)
                        .ok()
                        .map(|(_, f)| f)
                });
                match field {
                    Some(f) => {
                        f.data_type().hash(&mut h);
                        f.is_nullable().hash(&mut h);
                    }
                    None => 0u8.hash(&mut h),
                }
            }
            out.push(h.finish());
            Ok(TreeNodeRecursion::Continue)
        });
        Ok(TreeNodeRecursion::Continue)
    });
    out
}

/// Fingerprints of every subexpression of every expression (same key as
/// `simplify_expr_fingerprints`).
fn simplify_subexpr_fingerprints(plan: &LogicalPlan) -> Vec<u64> {
    let mut out = vec![];
    let _ = plan.apply_with_subqueries(|n| {
        let inputs = n.inputs();
        let is_join = matches!(n, LogicalPlan::Join(_));
        let _ = n.apply_expressions(|e| {
            e.apply(|x| {
                let mut h = DefaultHasher::new();
                is_join.hash(&mut h);
                x.hash(&mut h);
                for c in sorted_refs(x) {
                    let field = inputs.iter().find_map(|i| {
                        i.schema()
                            .qualified_field_from_column(c)
                            .ok()
                            .map(|(_, f)| f)
                    });
                    match field {
                        Some(f) => {
                            f.data_type().hash(&mut h);
                            f.is_nullable().hash(&mut h);
                        }
                        None => 0u8.hash(&mut h),
                    }
                }
                out.push(h.finish());
                Ok(TreeNodeRecursion::Continue)
            })
        });
        Ok(TreeNodeRecursion::Continue)
    });
    out
}

/// What `simplify_expressions` looks at for one node: the node kind, its
/// expressions and its input schemas.
fn simplify_fingerprints(plan: &LogicalPlan) -> Vec<u64> {
    let mut out = vec![];
    let _ = plan.apply_with_subqueries(|n| {
        let mut h = DefaultHasher::new();
        std::mem::discriminant(n).hash(&mut h);
        let _ = n.apply_expressions(|e| {
            e.hash(&mut h);
            Ok(TreeNodeRecursion::Continue)
        });
        for i in n.inputs() {
            i.schema().hash(&mut h);
        }
        if let LogicalPlan::TableScan(t) = n {
            t.table_name.hash(&mut h);
        }
        out.push(h.finish());
        Ok(TreeNodeRecursion::Continue)
    });
    out
}

fn any_node(plan: &LogicalPlan, f: impl Fn(&LogicalPlan) -> bool) -> bool {
    let mut found = false;
    plan.apply_with_subqueries(|n| {
        if f(n) {
            found = true;
            Ok(TreeNodeRecursion::Stop)
        } else {
            Ok(TreeNodeRecursion::Continue)
        }
    })
    .unwrap();
    found
}

fn any_expr(plan: &LogicalPlan, f: impl Fn(&Expr) -> bool) -> bool {
    let mut found = false;
    plan.apply_with_subqueries(|n| {
        n.apply_expressions(|e| {
            e.apply(|x| {
                if f(x) {
                    found = true;
                    Ok(TreeNodeRecursion::Stop)
                } else {
                    Ok(TreeNodeRecursion::Continue)
                }
            })
        })
    })
    .unwrap();
    found
}

fn has_distinct_agg(exprs: &[Expr]) -> bool {
    exprs.iter().any(|e| {
        e.exists(|x| Ok(matches!(x, Expr::AggregateFunction(af) if af.params.distinct)))
            .unwrap()
    })
}

/// A cheap, conservative "might this rule change the plan?" check, derived from
/// the node/expression shapes each rule matches on. `None`: no cheap check.
fn trigger(rule: &str, plan: &LogicalPlan) -> Option<bool> {
    use LogicalPlan as P;
    Some(match rule {
        "rewrite_set_comparison" => {
            any_expr(plan, |e| matches!(e, Expr::SetComparison(_)))
        }
        "optimize_unions" => any_node(plan, |n| matches!(n, P::Union(_))),
        "replace_distinct_aggregate" => any_node(plan, |n| matches!(n, P::Distinct(_))),
        "eliminate_join" | "eliminate_cross_join" => {
            any_node(plan, |n| matches!(n, P::Join(_)))
        }
        "decorrelate_predicate_subquery" => any_expr(plan, |e| {
            matches!(
                e,
                Expr::Exists(_) | Expr::InSubquery(_) | Expr::SetComparison(_)
            )
        }),
        "scalar_subquery_to_join" => {
            any_expr(plan, |e| matches!(e, Expr::ScalarSubquery(_)))
        }
        "decorrelate_lateral_join" => any_node(plan, |n| matches!(n, P::Subquery(_))),
        "extract_equijoin_predicate" => {
            any_node(plan, |n| matches!(n, P::Join(j) if j.filter.is_some()))
        }
        "eliminate_duplicated_expr" => {
            any_node(plan, |n| matches!(n, P::Sort(_) | P::Aggregate(_)))
        }
        "eliminate_filter" => any_node(plan, |n| match n {
            P::Filter(f) => f
                .predicate
                .exists(|e| {
                    Ok(matches!(e, Expr::Literal(v, _)
                        if v.is_null() || matches!(v, ScalarValue::Boolean(_))))
                })
                .unwrap(),
            _ => false,
        }),
        "eliminate_limit" | "push_down_limit" => {
            any_node(plan, |n| matches!(n, P::Limit(_)))
        }
        "propagate_empty_relation" => {
            any_node(plan, |n| matches!(n, P::EmptyRelation(_)))
        }
        "filter_null_join_keys" => any_node(plan, |n| match n {
            P::Join(j) => j.on.iter().any(|(l, r)| {
                l.nullable(j.left.schema()).unwrap_or(true)
                    || r.nullable(j.right.schema()).unwrap_or(true)
            }),
            _ => false,
        }),
        "eliminate_outer_join" => any_node(
            plan,
            |n| matches!(n, P::Join(j) if matches!(j.join_type, JoinType::Left | JoinType::Right | JoinType::Full)),
        ),
        "eliminate_aggregate_distinct" | "single_distinct_aggregation_to_group_by" => {
            any_node(
                plan,
                |n| matches!(n, P::Aggregate(a) if has_distinct_agg(&a.aggr_expr)),
            )
        }
        "eliminate_group_by_constant" => any_node(plan, |n| {
            matches!(n, P::Aggregate(a) if a.group_expr.len() >= 2
                && a.group_expr.iter().any(|g| !matches!(g, Expr::Column(_))))
        }),
        "extract_leaf_expressions" | "push_down_leaf_projections" => {
            any_expr(plan, |e| {
                e.placement() == ExpressionPlacement::MoveTowardsLeafNodes
            })
        }
        "replace_filter_top1" => any_node(plan, |n| matches!(n, P::Window(_))),
        _ => return None,
    })
}

/// Runs the logical optimizer once, recording time and change per rule
/// invocation. Time for the observer itself (clone + compare) is excluded.
fn observe_logical(state: &SessionState, plan: &LogicalPlan) -> Result<Vec<LogRecord>> {
    let n_rules = state.optimizer().rules.len();
    let mut prev = plan.clone();
    let mut records = vec![];
    let mut idx = 0usize;
    let mut simplified: HashSet<u64> = HashSet::new();
    let mut simplified_exprs: HashSet<u64> = HashSet::new();
    let mut simplified_sub: HashSet<u64> = HashSet::new();
    let mut last_end = Instant::now();
    state
        .optimizer()
        .optimize(plan.clone(), state, |new, rule| {
            let dur = last_end.elapsed();
            let t = Instant::now();
            let trig = trigger(rule.name(), &prev);
            let trigger_dur = t.elapsed();
            let t = Instant::now();
            let (cse_dup, simp_dirty) = match rule.name() {
                "common_sub_expression_eliminate" => (Some(cse_might_fire(&prev)), None),
                "simplify_expressions" if !simplified.is_empty() => {
                    let fps = simplify_fingerprints(&prev);
                    let dirty = fps.iter().filter(|f| !simplified.contains(f)).count();
                    let efps = simplify_expr_fingerprints(&prev);
                    let edirty = efps
                        .iter()
                        .filter(|f| !simplified_exprs.contains(f))
                        .count();
                    let sdirty =
                        efps.iter().filter(|f| !simplified_sub.contains(f)).count();
                    (None, Some((dirty, fps.len(), edirty, efps.len(), sdirty)))
                }
                _ => (None, None),
            };
            let prop_dur = t.elapsed();
            let changed = *new != prev;
            if rule.name() == "simplify_expressions" {
                simplified.extend(simplify_fingerprints(new));
                simplified_exprs.extend(simplify_expr_fingerprints(new));
                simplified_sub.extend(simplify_subexpr_fingerprints(new));
            }
            if changed {
                prev = new.clone();
            }
            records.push(LogRecord {
                pass: idx / n_rules,
                name: rule.name().to_string(),
                dur,
                changed,
                trigger: trig,
                trigger_dur,
                cse_dup,
                simp_dirty,
                prop_dur,
            });
            idx += 1;
            last_end = Instant::now();
        })?;
    Ok(records)
}

async fn make_ctx(args: &Args) -> Result<SessionContext> {
    let config = SessionConfig::new().with_target_partitions(args.partitions);
    let ctx = SessionContext::new_with_config(config);
    for table in TPCH_TABLES {
        match &args.data {
            Some(dir) => {
                let file = format!("{dir}/{table}.parquet");
                let path = if std::path::Path::new(&file).exists() {
                    file
                } else {
                    format!("{dir}/{table}/")
                };
                ctx.register_parquet(*table, &path, ParquetReadOptions::default())
                    .await?;
            }
            None => {
                let schema = Arc::new(get_tpch_table_schema(table));
                let mem = MemTable::try_new(schema, vec![vec![]])?;
                ctx.register_table(*table, Arc::new(mem))?;
            }
        }
    }
    Ok(ctx)
}

fn with_physical_rules(
    state: &SessionState,
    rules: Vec<Arc<dyn PhysicalOptimizerRule + Send + Sync>>,
) -> SessionState {
    SessionStateBuilder::new_from_existing(state.clone())
        .with_physical_optimizer_rules(rules)
        .build()
}

async fn physical_from_optimized(
    state: &SessionState,
    plan: &LogicalPlan,
) -> Result<Arc<dyn ExecutionPlan>> {
    state
        .query_planner()
        .create_physical_plan(plan, state)
        .await
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let args = Args::parse();
    let ctx = make_ctx(&args).await?;
    let iters = args.iterations;
    let alias_re = regex::Regex::new(r"(__[a-z_]+?)_\d+").unwrap();

    let queries: Vec<usize> = match args.profile_query {
        Some(q) => vec![q],
        None => (1..=22).collect(),
    };
    for q in queries {
        let sql = std::fs::read_to_string(format!("{}/q{q}.sql", args.queries))?;
        let stmts: Vec<&str> = sql
            .split(';')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        // q15: create view / select / drop view
        let (setup, main_sql, teardown): (Vec<&str>, &str, Vec<&str>) =
            if stmts.len() == 3 {
                (vec![stmts[0]], stmts[1], vec![stmts[2]])
            } else {
                (vec![], stmts[0], vec![])
            };
        for s in &setup {
            ctx.sql(s).await?.collect().await?;
        }

        let state = ctx.state();
        let raw = state.create_logical_plan(main_sql).await?;
        let analyzed = state.analyzer().execute_and_check(
            raw,
            state.config().options(),
            |_, _| {},
        )?;

        if args.profile_query.is_some() {
            let optimized = state.optimizer().optimize(analyzed, &state, |_, _| {})?;
            for _ in 0..args.loops {
                physical_from_optimized(&state, &optimized).await?;
            }
            if args.loops == 0 {
                let p = physical_from_optimized(&state, &optimized).await?;
                println!("{}", displayable(p.as_ref()).indent(false));
            }
            for s in &teardown {
                ctx.sql(s).await?.collect().await?;
            }
            continue;
        }

        // ---------- logical ----------
        let mut runs: Vec<Vec<LogRecord>> = vec![];
        for _ in 0..iters {
            runs.push(observe_logical(&state, &analyzed)?);
        }
        let n = runs[0].len();
        for r in &runs {
            assert_eq!(r.len(), n);
        }
        let mut changed_rules: Vec<Arc<dyn OptimizerRule + Send + Sync>> = vec![];
        let all_rules = state.optimizer().rules.clone();
        for i in 0..n {
            let rec = &runs[0][i];
            let dur = median(runs.iter().map(|r| r[i].dur).collect());
            let trig_dur = median(runs.iter().map(|r| r[i].trigger_dur).collect());
            let prop_dur = median(runs.iter().map(|r| r[i].prop_dur).collect());
            if rec.changed {
                changed_rules.push(Arc::clone(&all_rules[i % all_rules.len()]));
            }
            println!(
                "{}",
                json!({"q": q, "phase": "logical", "pass": rec.pass, "rule": rec.name,
                       "ns": dur.as_nanos() as u64, "changed": rec.changed,
                       "trigger": rec.trigger, "trigger_ns": trig_dur.as_nanos() as u64,
                       "cse_dup": rec.cse_dup, "simp_dirty": rec.simp_dirty,
                       "prop_ns": prop_dur.as_nanos() as u64})
            );
        }

        // full optimizer time (no observer)
        let full_times: Vec<Duration> = (0..iters)
            .map(|_| {
                let t = Instant::now();
                let _ = state
                    .optimizer()
                    .optimize(analyzed.clone(), &state, |_, _| {});
                t.elapsed()
            })
            .collect();
        let optimized =
            state
                .optimizer()
                .optimize(analyzed.clone(), &state, |_, _| {})?;

        // oracle: only the changing invocations, in order, single pass
        let oracle_state = {
            // keep the shared catalog: a config with create_default_catalog=true
            // would replace it on build()
            let mut cfg = state
                .config()
                .clone()
                .with_create_default_catalog_and_schema(false);
            cfg.options_mut().optimizer.max_passes = 1;
            SessionStateBuilder::new_from_existing(state.clone())
                .with_config(cfg)
                .with_optimizer_rules(changed_rules.clone())
                .build()
        };
        let oracle_opt = Optimizer::with_rules(changed_rules.clone());
        let oracle_times: Vec<Duration> = (0..iters)
            .map(|_| {
                let t = Instant::now();
                let _ = oracle_opt.optimize(analyzed.clone(), &oracle_state, |_, _| {});
                t.elapsed()
            })
            .collect();
        let oracle_plan =
            oracle_opt.optimize(analyzed.clone(), &oracle_state, |_, _| {})?;
        // generated aliases (__common_expr_N, __scalar_sq_N) depend on a
        // session-wide counter, so compare with the numbers stripped
        let norm = |p: &LogicalPlan| {
            alias_re
                .replace_all(&p.display_indent_schema().to_string(), "${1}_N")
                .to_string()
        };
        let logical_same = norm(&oracle_plan) == norm(&optimized);
        if !logical_same && std::env::var("DUMP").is_ok() {
            eprintln!(
                "q{q} FULL:\n{}\nORACLE:\n{}",
                optimized.display_indent_schema(),
                oracle_plan.display_indent_schema()
            );
        }

        // ---------- physical ----------
        let orig_rules = state.physical_optimizers().to_vec();
        let log = Arc::new(Mutex::new(Vec::new()));
        let wrapped: Vec<Arc<dyn PhysicalOptimizerRule + Send + Sync>> = orig_rules
            .iter()
            .map(|r| {
                Arc::new(Recording {
                    inner: Arc::clone(r),
                    log: Arc::clone(&log),
                }) as _
            })
            .collect();
        let rec_state = with_physical_rules(&state, wrapped);
        let mut phys_runs: Vec<Vec<PhysRecord>> = vec![];
        for _ in 0..iters {
            log.lock().unwrap().clear();
            physical_from_optimized(&rec_state, &optimized).await?;
            phys_runs.push(log.lock().unwrap().clone());
        }
        let mut phys_changed_rules = vec![];
        for (i, rule) in orig_rules.iter().enumerate() {
            let rec = &phys_runs[0][i];
            let dur = median(phys_runs.iter().map(|r| r[i].dur).collect());
            if rec.changed {
                phys_changed_rules.push(Arc::clone(rule));
            }
            println!(
                "{}",
                json!({"q": q, "phase": "physical", "pass": 0, "rule": rec.name,
                       "ns": dur.as_nanos() as u64, "changed": rec.changed,
                       "ptr_same": rec.ptr_same, "nodes": rec.rebuild.nodes,
                       "new_nodes": rec.rebuild.new_nodes, "needless_new": rec.rebuild.needless,
                       "same_props": rec.rebuild.same_props,
                       "same_props_kids": rec.rebuild.same_props_and_child_props,
                       "new_by_type": rec.rebuild.new_by_type})
            );
        }

        // Full physical planning (initial plan + all rules) vs oracle subset.
        let mut phys_full = vec![];
        let mut phys_oracle = vec![];
        let oracle_pstate = with_physical_rules(&state, phys_changed_rules);
        let none_pstate = with_physical_rules(&state, vec![]);
        let mut phys_none = vec![];
        for _ in 0..iters {
            let t = Instant::now();
            physical_from_optimized(&state, &optimized).await?;
            phys_full.push(t.elapsed());
            let t = Instant::now();
            physical_from_optimized(&oracle_pstate, &optimized).await?;
            phys_oracle.push(t.elapsed());
            let t = Instant::now();
            // may fail the sanity check without rules; only want the time
            let _ = physical_from_optimized(&none_pstate, &optimized).await;
            phys_none.push(t.elapsed());
        }
        let full_p = physical_from_optimized(&state, &optimized).await?;
        let oracle_p = physical_from_optimized(&oracle_pstate, &optimized).await?;
        let physical_same = phys_string(&full_p) == phys_string(&oracle_p);
        let plan_hash = {
            let mut h = DefaultHasher::new();
            phys_string(&full_p).hash(&mut h);
            h.finish()
        };

        println!(
            "{}",
            json!({"q": q, "phase": "summary",
                "logical_full_ns": median(full_times).as_nanos() as u64,
                "logical_oracle_ns": median(oracle_times).as_nanos() as u64,
                "logical_oracle_same_plan": logical_same,
                "physical_full_ns": median(phys_full).as_nanos() as u64,
                "physical_oracle_ns": median(phys_oracle).as_nanos() as u64,
                "physical_no_rules_ns": median(phys_none).as_nanos() as u64,
                "physical_oracle_same_plan": physical_same,
                "physical_plan_hash": plan_hash,
            })
        );

        for s in &teardown {
            ctx.sql(s).await?.collect().await?;
        }
    }
    Ok(())
}
