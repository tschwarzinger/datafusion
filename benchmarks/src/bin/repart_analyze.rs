//! Throwaway analysis tool (not for commit): for each query, list hash
//! RepartitionExecs and whether their input is sorted on all hash keys.
//!
//! usage: repart_analyze <setup.sql> <query files...>
//! Each query file may contain several `;`-separated statements; all but the
//! last are executed, the last one is planned.

use std::sync::Arc;

use datafusion::error::Result;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::{
    ExecutionPlan, ExecutionPlanProperties, Partitioning, displayable,
};
use datafusion::prelude::*;

fn split_statements(sql: &str) -> Vec<String> {
    let mut out = vec![];
    let mut cur = String::new();
    let mut in_str = false;
    for line in sql.lines() {
        let t = line.trim_start();
        if !in_str && t.starts_with("--") {
            continue;
        }
        for c in line.chars() {
            if c == '\'' {
                in_str = !in_str;
            }
            if c == ';' && !in_str {
                if !cur.trim().is_empty() {
                    out.push(cur.trim().to_string());
                }
                cur.clear();
            } else {
                cur.push(c);
            }
        }
        cur.push('\n');
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

fn walk(plan: &Arc<dyn ExecutionPlan>, query: &str, found: &mut usize) -> Result<()> {
    if let Some(r) = plan.downcast_ref::<RepartitionExec>()
        && let Partitioning::Hash(exprs, n) = r.partitioning()
    {
        let input = r.input();
        let eq = input.equivalence_properties();
        let (perm, _) = eq.find_longest_permutation(exprs)?;
        let ordering = input
            .output_ordering()
            .map(|o| o.to_string())
            .unwrap_or_else(|| "-".to_string());
        let keys = exprs
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let status = if perm.len() == exprs.len() {
            "FULL"
        } else if perm.is_empty() {
            "none"
        } else {
            "PARTIAL"
        };
        let child = input.name();
        let in_parts = input.output_partitioning().partition_count();
        println!(
            "{query}\t{status}\tkeys=[{keys}]\tparts={in_parts}->{n}\tpreserve_order={}\tinput={child}\tinput_ordering=[{ordering}]",
            r.preserve_order()
        );
        *found += 1;
    }
    for c in plan.children() {
        walk(c, query, found)?;
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let config = SessionConfig::from_env()?;
    let ctx = SessionContext::new_with_config(config);
    let setup = std::fs::read_to_string(&args[1])?;
    for stmt in split_statements(&setup) {
        ctx.sql(&stmt).await?.collect().await?;
    }
    let show_plans = std::env::var("SHOW_PLANS").is_ok();
    for qf in &args[2..] {
        let name = std::path::Path::new(qf)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let stmts = split_statements(&std::fs::read_to_string(qf)?);
        let Some((last, pre)) = stmts.split_last() else {
            continue;
        };
        let res: Result<()> = async {
            for s in pre {
                ctx.sql(s).await?.collect().await?;
            }
            let df = ctx.sql(last).await?;
            let plan = df.create_physical_plan().await?;
            if show_plans {
                eprintln!("=== {name}\n{}", displayable(plan.as_ref()).indent(true));
            }
            if std::env::var("RUN_ANALYZE").is_ok() {
                let task_ctx = ctx.task_ctx();
                let start = datafusion_common::instant::Instant::now();
                let batches =
                    datafusion::physical_plan::collect(Arc::clone(&plan), task_ctx)
                        .await?;
                let elapsed = start.elapsed();
                let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
                println!("{name}\telapsed_ms={}\trows={rows}", elapsed.as_millis());
                if std::env::var("CHECKSUM").is_ok() {
                    use std::hash::{Hash, Hasher};
                    let formatted = arrow::util::pretty::pretty_format_batches(&batches)?
                        .to_string();
                    let mut lines: Vec<&str> = formatted
                        .lines()
                        .filter(|l| l.starts_with("| "))
                        .skip(1)
                        .collect();
                    lines.sort_unstable();
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    lines.hash(&mut h);
                    println!("{name}\tchecksum={:x}", h.finish());
                    return Ok(());
                }
                println!(
                    "{}",
                    datafusion::physical_plan::display::DisplayableExecutionPlan::with_metrics(
                        plan.as_ref()
                    )
                    .indent(true)
                );
                return Ok(());
            }
            let mut found = 0;
            walk(&plan, &name, &mut found)?;
            if found == 0 {
                println!("{name}\tNO_HASH_REPARTITION");
            }
            Ok(())
        }
        .await;
        if let Err(e) = res {
            println!("{name}\tERROR\t{}", e.to_string().replace('\n', " "));
        }
    }
    Ok(())
}
