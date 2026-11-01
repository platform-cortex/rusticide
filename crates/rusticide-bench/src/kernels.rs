// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::{Duration, Instant};

use rusticide_quantize::{BlockFormat, Q8Input, Q8Kind, formats};

use crate::{BenchError, Plan, Sample, throughput};

const ROWS: usize = 256;
const COLUMNS: usize = 4096;
const BATCH: usize = 8;
const ROW_WORK: f64 = 256.0 * 4096.0;

/// Deterministic values in `[-1, 1)`.
fn activations(seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..COLUMNS)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f32::from(u16::try_from(state >> 16).unwrap_or(0)) / 32768.0 - 1.0
        })
        .collect()
}

fn quantized(kind: Q8Kind, seed: u32) -> Q8Input {
    let mut input = Q8Input::default();
    input.quantize(&activations(seed), kind);
    input
}

/// Repeats `pass` until `budget` elapses and returns work units per second.
fn sustained(
    budget: Duration,
    work_per_pass: f64,
    mut pass: impl FnMut(),
) -> Result<f64, BenchError> {
    let start = Instant::now();
    let mut passes = 0.0;
    while start.elapsed() < budget {
        pass();
        passes += 1.0;
    }
    throughput(work_per_pass * passes, start.elapsed())
}

fn row_metric(format: &BlockFormat, plan: &Plan) -> Result<Sample, BenchError> {
    let matrix = format.synthetic_blocks(ROWS * COLUMNS / format.block_size(), 1);
    let row_bytes = format.row_bytes(COLUMNS).unwrap_or(matrix.len());
    let input = quantized(format.activation(), 2);
    plan.sample(|| {
        sustained(plan.budget, ROW_WORK, || {
            for row in matrix.chunks_exact(row_bytes) {
                black_box(format.dot_row_q8(black_box(row), black_box(&input)));
            }
        })
    })
}

fn rows8_metric(format: &BlockFormat, plan: &Plan) -> Result<Sample, BenchError> {
    let matrix = format.synthetic_blocks(ROWS * COLUMNS / format.block_size(), 1);
    let row_bytes = format.row_bytes(COLUMNS).unwrap_or(matrix.len());
    let inputs: Vec<Q8Input> = (3..3 + 8)
        .map(|seed| quantized(format.activation(), seed))
        .collect();
    let mut out = [0.0_f32; BATCH];
    plan.sample(|| {
        sustained(plan.budget, ROW_WORK * 8.0, || {
            for row in matrix.chunks_exact(row_bytes) {
                format.dot_rows_q8(black_box(row), black_box(&inputs), &mut out);
                black_box(&out);
            }
        })
    })
}

fn quantize_metric(kind: Q8Kind, plan: &Plan) -> Result<Sample, BenchError> {
    let values = activations(4);
    let mut input = Q8Input::default();
    plan.sample(|| {
        sustained(plan.budget, 4096.0, || {
            input.quantize(black_box(&values), kind);
            black_box(&input);
        })
    })
}

/// Single-thread kernel throughputs for every block format and both
/// activation quantizers.
///
/// # Errors
///
/// A failure to summarize the runs.
pub fn kernel_metrics(plan: &Plan) -> Result<BTreeMap<String, Sample>, BenchError> {
    let mut metrics = BTreeMap::new();
    for format in formats() {
        let name = format.name();
        metrics.insert(format!("kernel/{name}/row"), row_metric(format, plan)?);
        metrics.insert(format!("kernel/{name}/rows8"), rows8_metric(format, plan)?);
    }
    metrics.insert(
        "kernel/q8/block".into(),
        quantize_metric(Q8Kind::Block, plan)?,
    );
    metrics.insert(
        "kernel/q8/superblock".into(),
        quantize_metric(Q8Kind::SuperBlock, plan)?,
    );
    Ok(metrics)
}

#[cfg(test)]
#[path = "kernels_tests.rs"]
mod tests;
