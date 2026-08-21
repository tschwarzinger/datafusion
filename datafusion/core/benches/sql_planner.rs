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

mod data_utils;

use arrow::array::PrimitiveArray;
use arrow::array::{ArrayRef, RecordBatch};
use arrow::datatypes::ArrowNativeTypeOp;
use arrow::datatypes::ArrowPrimitiveType;
use arrow::datatypes::{DataType, Field, Fields, Schema};
use criterion::Bencher;
use criterion::{Criterion, criterion_group, criterion_main};
use datafusion::datasource::MemTable;
use datafusion::execution::context::SessionContext;
use datafusion_common::{ScalarValue, config::Dialect};
use datafusion_expr::col;
use rand_distr::num_traits::NumCast;
use std::hint::black_box;
use std::path::PathBuf;
use std::sync::Arc;
use test_utils::TableDef;
use test_utils::tpch::tpch_schemas;
use tokio::runtime::Runtime;

const BENCHMARKS_PATH_1: &str = "../../benchmarks/";
const BENCHMARKS_PATH_2: &str = "./benchmarks/";
const CLICKBENCH_DATA_PATH: &str = "data/hits_partitioned/";

/// Create a logical plan from the specified sql (parse + analyze only, NO optimization)
fn logical_plan(ctx: &SessionContext, rt: &Runtime, sql: &str) {
    black_box(rt.block_on(ctx.sql(sql)).unwrap());
}

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

/// Create a physical ExecutionPlan (by way of logical plan)
fn physical_plan(ctx: &SessionContext, rt: &Runtime, sql: &str) {
    black_box(rt.block_on(async {
        ctx.sql(sql)
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap()
    }));
}

/// Create schema with the specified number of columns
fn create_schema(column_prefix: &str, num_columns: usize) -> Schema {
    let fields: Fields = (0..num_columns)
        .map(|i| Field::new(format!("{column_prefix}{i}"), DataType::Int32, true))
        .collect();
    Schema::new(fields)
}

fn create_table_provider(column_prefix: &str, num_columns: usize) -> Arc<MemTable> {
    let schema = Arc::new(create_schema(column_prefix, num_columns));
    MemTable::try_new(schema, vec![vec![]])
        .map(Arc::new)
        .unwrap()
}

/// Create a table provider with a struct column: `id` (Int32) and `props` (Struct { value: Int32, label: Utf8 })
fn create_struct_table_provider() -> Arc<MemTable> {
    let struct_fields = Fields::from(vec![
        Field::new("value", DataType::Int32, true),
        Field::new("label", DataType::Utf8, true),
    ]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, true),
        Field::new("props", DataType::Struct(struct_fields), true),
    ]));
    MemTable::try_new(schema, vec![vec![]])
        .map(Arc::new)
        .unwrap()
}

fn create_context() -> SessionContext {
    let ctx = SessionContext::new();
    ctx.register_table("t1", create_table_provider("a", 200))
        .unwrap();
    ctx.register_table("t2", create_table_provider("b", 200))
        .unwrap();
    ctx.register_table("t700", create_table_provider("c", 700))
        .unwrap();
    ctx.register_table("t1000", create_table_provider("d", 1000))
        .unwrap();
    ctx.register_table("struct_t1", create_struct_table_provider())
        .unwrap();
    ctx.register_table("struct_t2", create_struct_table_provider())
        .unwrap();
    ctx
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

/// Target of this benchmark: control that placeholders replacing does not get slower,
/// if the query does not contain placeholders at all.
fn benchmark_with_param_values_many_columns(
    ctx: &SessionContext,
    rt: &Runtime,
    b: &mut Bencher,
) {
    const COLUMNS_NUM: usize = 200;
    let mut aggregates = String::new();
    for i in 0..COLUMNS_NUM {
        if i > 0 {
            aggregates.push_str(", ");
        }
        aggregates.push_str(format!("MAX(a{i})").as_str());
    }
    // SELECT max(attr0), ..., max(attrN) FROM t1.
    let query = format!("SELECT {aggregates} FROM t1");
    let statement = ctx
        .state()
        .sql_to_statement(&query, &Dialect::Generic)
        .unwrap();
    let plan =
        rt.block_on(async { ctx.state().statement_to_plan(statement).await.unwrap() });
    b.iter(|| {
        let plan = plan.clone();
        black_box(plan.with_param_values(vec![ScalarValue::from(1)]).unwrap());
    });
}

/// Registers a table like this:
/// c0,c1,c2...,c99
/// 0,100...9900
/// 0,200...19800
/// 0,300...29700
fn register_union_order_table_generic<T>(
    ctx: &SessionContext,
    num_columns: usize,
    num_rows: usize,
) where
    T: ArrowPrimitiveType,
    T::Native: ArrowNativeTypeOp + NumCast,
{
    let iter = (0..num_columns).map(|i| {
        let array_data: Vec<T::Native> = (0..num_rows)
            .map(|j| {
                let value = (j as u64) * 100 + (i as u64);
                <T::Native as NumCast>::from(value).unwrap_or_else(|| {
                    panic!("Failed to cast numeric value to Native type")
                })
            })
            .collect();

        // Use PrimitiveArray which is generic over the ArrowPrimitiveType T
        let array: ArrayRef = Arc::new(PrimitiveArray::<T>::from_iter_values(array_data));

        (format!("c{i}"), array)
    });

    let batch = RecordBatch::try_from_iter(iter).unwrap();
    let schema = batch.schema();
    let partitions = vec![vec![batch]];

    // tell DataFusion that the table is sorted by all columns
    let sort_order = (0..num_columns)
        .map(|i| col(format!("c{i}")).sort(true, true))
        .collect::<Vec<_>>();

    // create the table
    let table = MemTable::try_new(schema, partitions)
        .unwrap()
        .with_sort_order(vec![sort_order]);

    ctx.register_table("t", Arc::new(table)).unwrap();
}
/// return a query like
/// ```sql
/// select c1, 2 as c2, ... n as cn from t ORDER BY c1
///   UNION ALL
/// select 1 as c1, c2, ... n as cn from t ORDER BY c2
/// ...
/// select 1 as c1, 2 as c2, ... cn from t ORDER BY cn
///  ORDER BY c1, c2 ... CN
/// ```
fn union_orderby_query(n: usize) -> String {
    let mut query = String::new();
    for i in 0..n {
        if i != 0 {
            query.push_str("\n  UNION ALL \n");
        }
        let select_list = (0..n)
            .map(|j| {
                if i == j {
                    format!("c{j}")
                } else {
                    format!("{j} as c{j}")
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        query.push_str(&format!("(SELECT {select_list} FROM t ORDER BY c{i})"));
    }
    query.push_str(&format!(
        "\nORDER BY {}",
        (0..n)
            .map(|i| format!("c{i}"))
            .collect::<Vec<_>>()
            .join(", ")
    ));
    query
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

    let ctx = create_context();
    let rt = Runtime::new().unwrap();

    // --- TPC-H ---

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

    // -- clickbench --
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

    // TPC-H all queries (optimizer only)
    {
        let plans: Vec<_> = all_tpch_sql_queries
            .iter()
            .map(|sql| analyzed_plan(&tpch_ctx, &rt, sql))
            .collect();
        for plan in &plans {
            optimize_plan(&tpch_ctx, plan)
        }
    }

    // Clickbench all queries (optimizer only)
    {
        let plans: Vec<_> = clickbench_queries
            .iter()
            .map(|sql| analyzed_plan(&clickbench_ctx, &rt, sql))
            .collect();
        for plan in &plans {
            optimize_plan(&clickbench_ctx, plan)
        }
    }

}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
