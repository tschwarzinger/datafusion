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

use criterion::{Criterion, criterion_group, criterion_main};
use datafusion::datasource::MemTable;
use datafusion::execution::context::SessionContext;
use std::hint::black_box;
use std::path::PathBuf;
use std::sync::Arc;
use test_utils::TableDef;
use test_utils::tpcds::tpcds_schemas;
use test_utils::tpch::tpch_schemas;
use tokio::runtime::Runtime;

const BENCHMARKS_PATH_1: &str = "../../benchmarks/";
const BENCHMARKS_PATH_2: &str = "./benchmarks/";
const CLICKBENCH_DATA_PATH: &str = "data/hits_partitioned/";

/// Parse SQL and run the analyzer to get an analyzed (but unoptimized) LogicalPlan.
/// This is the input to the optimizer.
fn analyzed_plan(
    ctx: &SessionContext,
    rt: &Runtime,
    sql: &str,
) -> datafusion_expr::LogicalPlan {
    let state = ctx.state();
    let plan = rt.block_on(state.create_logical_plan(sql)).unwrap();
    state
        .analyzer()
        .execute_and_check(plan, state.config().options(), |_, _| {})
        .unwrap()
}

/// Run ONLY the optimizer on a pre-analyzed plan. Measures optimizer cost in isolation.
fn optimize_plan(ctx: &SessionContext, plan: &datafusion_expr::LogicalPlan) {
    let state = ctx.state();
    black_box(
        state
            .optimizer()
            .optimize(plan.clone(), &state, |_, _| {})
            .unwrap(),
    );
}

/// Register the table definitions as a MemTable with the context and return the
/// context
fn register_defs(ctx: SessionContext, defs: Vec<TableDef>) -> SessionContext {
    for TableDef {
        name,
        schema,
        constraints,
    } in defs
    {
        ctx.register_table(
            &name,
            Arc::new(
                MemTable::try_new(Arc::new(schema), vec![vec![]])
                    .unwrap()
                    .with_constraints(constraints),
            ),
        )
        .unwrap();
    }
    ctx
}

fn register_clickbench_hits_table(rt: &Runtime) -> SessionContext {
    let ctx = SessionContext::new();

    // use an external table for clickbench benchmarks
    let path =
        if PathBuf::from(format!("{BENCHMARKS_PATH_1}{CLICKBENCH_DATA_PATH}")).exists() {
            format!("{BENCHMARKS_PATH_1}{CLICKBENCH_DATA_PATH}")
        } else {
            format!("{BENCHMARKS_PATH_2}{CLICKBENCH_DATA_PATH}")
        };

    let sql =
        format!("CREATE EXTERNAL TABLE hits_raw STORED AS PARQUET LOCATION '{path}'");

    // ClickBench partitioned dataset was written by an ancient version of pyarrow that
    // that wrote strings with the wrong logical type. To read it correctly, we must
    // automatically convert binary to string.
    rt.block_on(ctx.sql("SET datafusion.execution.parquet.binary_as_string  = true;"))
        .unwrap();
    rt.block_on(ctx.sql(&sql)).unwrap();

    // ClickBench stores EventDate as UInt16 (days since 1970-01-01). Create a view
    // that exposes it as SQL DATE so that queries comparing it with date literals
    // (e.g. "EventDate >= '2013-07-01'") work correctly during planning.
    rt.block_on(ctx.sql(
        "CREATE VIEW hits AS \
         SELECT * EXCEPT (\"EventDate\"), \
                CAST(CAST(\"EventDate\" AS INTEGER) AS DATE) AS \"EventDate\" \
         FROM hits_raw",
    ))
    .unwrap();

    let count =
        rt.block_on(async { ctx.table("hits").await.unwrap().count().await.unwrap() });
    assert!(count > 0);
    ctx
}

fn criterion_benchmark(c: &mut Criterion) {
    // verify that we can load the clickbench data prior to running the benchmark
    if !PathBuf::from(format!("{BENCHMARKS_PATH_1}{CLICKBENCH_DATA_PATH}")).exists()
        && !PathBuf::from(format!("{BENCHMARKS_PATH_2}{CLICKBENCH_DATA_PATH}")).exists()
    {
        panic!(
            "benchmarks/data/hits_partitioned/ could not be loaded. Please run \
         'benchmarks/bench.sh data clickbench_partitioned' prior to running this benchmark"
        )
    }

    let rt = Runtime::new().unwrap();

    // --- TPC-H (optimizer only) ---

    let tpch_ctx = register_defs(SessionContext::new(), tpch_schemas());

    let tpch_queries = [
        "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13",
        "q14", // "q15", q15 has multiple SQL statements which is not supported
        "q16", "q17", "q18", "q19", "q20", "q21", "q22",
    ];

    let benchmarks_path = if PathBuf::from(BENCHMARKS_PATH_1).exists() {
        BENCHMARKS_PATH_1
    } else {
        BENCHMARKS_PATH_2
    };

    let all_tpch_sql_queries = tpch_queries
        .iter()
        .map(|q| {
            std::fs::read_to_string(format!("{benchmarks_path}queries/{q}.sql")).unwrap()
        })
        .collect::<Vec<_>>();

    {
        let plans: Vec<_> = all_tpch_sql_queries
            .iter()
            .map(|sql| analyzed_plan(&tpch_ctx, &rt, sql))
            .collect();
        c.bench_function("optimizer_tpch_all", |b| {
            b.iter(|| {
                for plan in &plans {
                    optimize_plan(&tpch_ctx, plan)
                }
            })
        });
    }

    // --- TPC-DS (optimizer only) ---

    let tpcds_ctx = register_defs(SessionContext::new(), tpcds_schemas());
    let tests_path = if PathBuf::from("./tests/").exists() {
        "./tests/"
    } else {
        "datafusion/core/tests/"
    };

    let raw_tpcds_sql_queries = (1..100)
        .map(|q| std::fs::read_to_string(format!("{tests_path}tpc-ds/{q}.sql")).unwrap())
        .collect::<Vec<_>>();

    // some queries have multiple statements
    let all_tpcds_sql_queries = raw_tpcds_sql_queries
        .iter()
        .flat_map(|sql| sql.split(';').filter(|s| !s.trim().is_empty()))
        .collect::<Vec<_>>();

    {
        let plans: Vec<_> = all_tpcds_sql_queries
            .iter()
            .map(|sql| analyzed_plan(&tpcds_ctx, &rt, sql))
            .collect();
        c.bench_function("optimizer_tpcds_all", |b| {
            b.iter(|| {
                for plan in &plans {
                    optimize_plan(&tpcds_ctx, plan)
                }
            })
        });
    }

    // --- ClickBench (optimizer only) ---

    let clickbench_queries = (0..=42)
        .map(|q| {
            std::fs::read_to_string(format!(
                "{benchmarks_path}queries/clickbench/queries/q{q}.sql"
            ))
            .unwrap()
        })
        .chain((0..=7).map(|q| {
            std::fs::read_to_string(format!(
                "{benchmarks_path}queries/clickbench/extended/q{q}.sql"
            ))
            .unwrap()
        }))
        .collect::<Vec<_>>();

    let clickbench_ctx = register_clickbench_hits_table(&rt);

    {
        let plans: Vec<_> = clickbench_queries
            .iter()
            .map(|sql| analyzed_plan(&clickbench_ctx, &rt, sql))
            .collect();
        c.bench_function("optimizer_clickbench_all", |b| {
            b.iter(|| {
                for plan in &plans {
                    optimize_plan(&clickbench_ctx, plan)
                }
            })
        });
    }
}
criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
