// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Benchmarks the row-group pruning a Parquet scan does when it opens a file.
//!
//! The files have many small row groups, so the cost of the scan is dominated
//! by building pruning predicates and evaluating them against the statistics,
//! Bloom filters and page indexes of the row groups, rather than by decoding.
//!
//! Every file has four `Int64` columns `g, p, o, s`, sorted by `g`. `s` is
//! spread over all row groups, so its statistics cannot prune, while most of
//! its values are present in every Bloom filter.

use std::hint::black_box;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use criterion::{Criterion, criterion_group, criterion_main};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;
use tokio::runtime::Runtime;

const NUM_FILES: usize = 4;
const ROW_GROUPS_PER_FILE: usize = 43;
const ROWS_PER_ROW_GROUP: usize = 256;
const DISTINCT_S: i64 = 1000;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(
        ["g", "p", "o", "s"]
            .into_iter()
            .map(|name| Field::new(name, DataType::Int64, false))
            .collect::<Vec<_>>(),
    ))
}

fn write_files(dir: &Path) {
    let schema = schema();
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(ROWS_PER_ROW_GROUP))
        .set_data_page_row_count_limit(64)
        .set_write_batch_size(64)
        .set_bloom_filter_enabled(true)
        .set_bloom_filter_max_ndv(ROWS_PER_ROW_GROUP as u64)
        .build();
    let rows_per_file = (ROW_GROUPS_PER_FILE * ROWS_PER_ROW_GROUP) as i64;
    for file in 0..NUM_FILES {
        let path = dir.join(format!("part-{file}.parquet"));
        let mut writer = ArrowWriter::try_new(
            std::fs::File::create(path).unwrap(),
            Arc::clone(&schema),
            Some(props.clone()),
        )
        .unwrap();
        let offset = file as i64 * rows_per_file;
        let g = Int64Array::from_iter_values((0..rows_per_file).map(|i| offset + i));
        let p = Int64Array::from_iter_values((0..rows_per_file).map(|i| i % 7));
        let o = Int64Array::from_iter_values(0..rows_per_file);
        let s = Int64Array::from_iter_values(
            (0..rows_per_file).map(|i| ((offset + i) * 7919) % DISTINCT_S),
        );
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(g), Arc::new(p), Arc::new(o), Arc::new(s)],
        )
        .unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }
}

fn in_list(column: &str, values: impl Iterator<Item = i64>) -> String {
    let values = values.map(|v| v.to_string()).collect::<Vec<_>>();
    format!("{column} IN ({})", values.join(", "))
}

/// Prints the pruning metrics of the scan, to see where the time goes.
fn print_metrics(rt: &Runtime, ctx: &SessionContext, name: &str, filter: &str) {
    let batches = rt
        .block_on(async {
            ctx.sql(&format!("EXPLAIN ANALYZE SELECT sum(o) FROM t {filter}"))
                .await?
                .collect()
                .await
        })
        .unwrap();
    let plan = arrow::util::pretty::pretty_format_batches(&batches)
        .unwrap()
        .to_string();
    for line in plan.lines().filter(|line| line.contains("DataSourceExec")) {
        for metric in line.split(", ").filter(|metric| {
            metric.contains("row_groups_pruned")
                || metric.contains("eval_time")
                || metric.contains("time_elapsed_opening")
                || metric.contains("page_index_rows")
        }) {
            eprintln!("{name}: {}", metric.trim());
        }
    }
}

fn parquet_open_pruning(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    write_files(dir.path());

    let rt = Runtime::new().unwrap();
    let config = SessionConfig::new()
        .with_target_partitions(1)
        .set_bool("datafusion.execution.parquet.pushdown_filters", true);
    let ctx = SessionContext::new_with_config(config);
    rt.block_on(ctx.register_parquet(
        "t",
        dir.path().to_str().unwrap(),
        ParquetReadOptions::default(),
    ))
    .unwrap();

    let total_rows = (NUM_FILES * ROW_GROUPS_PER_FILE * ROWS_PER_ROW_GROUP) as i64;
    let cases = [
        // Statistics cannot prune; Bloom filters prune about 1% of the row groups.
        (
            "in_list_unsorted",
            format!("WHERE {}", in_list("s", (0..16).map(|i| i * 61 + 3))),
        ),
        // Statistics prune all but a few row groups.
        (
            "in_list_sorted",
            format!(
                "WHERE {}",
                in_list("g", (0..16).map(|i| i * (total_rows / 16) + 5))
            ),
        ),
        // A range on the sort column that fully matches most surviving row groups.
        (
            "range_sorted",
            format!(
                "WHERE g >= {} AND g < {}",
                total_rows / 4,
                total_rows / 4 + 20 * ROWS_PER_ROW_GROUP as i64
            ),
        ),
        ("no_predicate", String::new()),
    ];

    let mut group = c.benchmark_group("parquet_open_pruning");
    for (name, filter) in cases {
        if std::env::var_os("PARQUET_OPEN_PRUNING_METRICS").is_some() {
            print_metrics(&rt, &ctx, name, &filter);
        }
        let query = format!("SELECT sum(o) FROM t {filter}");
        group.bench_function(name, |b| {
            b.iter(|| {
                let batches = rt
                    .block_on(async { ctx.sql(&query).await?.collect().await })
                    .unwrap();
                black_box(batches)
            })
        });
    }
    group.finish();
}

criterion_group!(benches, parquet_open_pruning);
criterion_main!(benches);
