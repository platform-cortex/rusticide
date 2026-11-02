// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_quantize::block_format;

use super::{TASKS_PER_THREAD, Weights, matmul_batch};
use crate::active;
use crate::test_support::{PAIRED_Q8_0, pseudo_random_values};

const COLS: usize = 64;

fn q8_0_matrix(rows: usize) -> (Vec<u8>, usize) {
    let format = block_format(8).unwrap();
    let data = format.synthetic_blocks(rows * COLS / 32, 11).unwrap();
    (data, format.row_bytes(COLS).unwrap())
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
    let mut prepared = original.clone();
    assert!(PAIRED_Q8_0.prepare.unwrap()(&mut prepared, row_bytes));
    let rows_kernel = active().quant_kernel(8).unwrap();
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
