// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_quantize::block_format;

use super::{TASKS_PER_THREAD, Weights, matmul_batch};
use crate::portable;
use crate::test_support::{PAIRED_Q8_0, pseudo_random_values};

const COLS: usize = 64;

fn q8_0_matrix(rows: usize) -> (Vec<u8>, usize) {
    let format = block_format(8).unwrap();
    let data = format.synthetic_blocks(rows * COLS / 32, 11).unwrap();
    (data, format.row_bytes(COLS).unwrap())
}

fn prepared(original: &[u8], row_bytes: usize) -> Vec<u8> {
    let mut data = original.to_vec();
    assert!((PAIRED_Q8_0.layout.unwrap().prepare)(&mut data, row_bytes));
    data
}

fn run(weights: Weights<'_>, inputs: &[f32], tokens: usize) -> Vec<u32> {
    let mut out = vec![f32::NAN; weights.rows() * tokens];
    matmul_batch(inputs, tokens, &mut [(weights, &mut out[..])]);
    out.into_iter().map(f32::to_bits).collect()
}

#[test]
fn grouped_kernels_match_row_kernels() {
    // An odd row count per task: unrounded, a run would start mid-group.
    let rows = rayon::current_num_threads() * TASKS_PER_THREAD * 2 + 2;
    let (original, row_bytes) = q8_0_matrix(rows);
    let prepared = prepared(&original, row_bytes);
    let rows_kernel = portable::ADAPTER.quant_kernel(8).unwrap();
    for tokens in [1, 3, 64] {
        let inputs = pseudo_random_values(tokens * COLS, 5);
        let expected = run(
            Weights::Blocks {
                kernel: rows_kernel,
                data: &original,
                row_bytes,
            },
            &inputs,
            tokens,
        );
        let grouped = Weights::Blocks {
            kernel: &PAIRED_Q8_0,
            data: &prepared,
            row_bytes,
        };
        assert_eq!(run(grouped, &inputs, tokens), expected, "{tokens} tokens");
    }
}

#[test]
fn a_trailing_partial_group_is_skipped() {
    let (data, row_bytes) = q8_0_matrix(3);
    let weights = Weights::Blocks {
        kernel: &PAIRED_Q8_0,
        data: &data,
        row_bytes,
    };
    let inputs = pseudo_random_values(COLS, 9);
    let mut out = [f32::NAN; 3];
    matmul_batch(&inputs, 1, &mut [(weights, &mut out[..])]);
    assert!(out[..2].iter().all(|value| value.is_finite()));
    assert!(out[2].is_nan());
}

#[test]
fn a_batch_mixes_grouped_and_row_major_jobs() {
    let rows = rayon::current_num_threads() * TASKS_PER_THREAD + 2;
    let (original, row_bytes) = q8_0_matrix(rows);
    let grouped_data = prepared(&original, row_bytes);
    let (row_major_data, _) = q8_0_matrix(rows + 1);
    let row_major = Weights::Blocks {
        kernel: portable::ADAPTER.quant_kernel(8).unwrap(),
        data: &row_major_data,
        row_bytes,
    };
    let grouped = Weights::Blocks {
        kernel: &PAIRED_Q8_0,
        data: &grouped_data,
        row_bytes,
    };
    let original_weights = Weights::Blocks {
        kernel: portable::ADAPTER.quant_kernel(8).unwrap(),
        data: &original,
        row_bytes,
    };
    let tokens = 3;
    let inputs = pseudo_random_values(tokens * COLS, 13);
    let (mut first, mut second) = (vec![0.0; rows * tokens], vec![0.0; (rows + 1) * tokens]);
    matmul_batch(
        &inputs,
        tokens,
        &mut [(grouped, &mut first[..]), (row_major, &mut second[..])],
    );
    let bits = |values: Vec<f32>| -> Vec<u32> { values.into_iter().map(f32::to_bits).collect() };
    assert_eq!(bits(first), run(original_weights, &inputs, tokens));
    assert_eq!(bits(second), run(row_major, &inputs, tokens));
}
