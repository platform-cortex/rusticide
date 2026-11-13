// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{TASKS_PER_THREAD, Weights, matmul_batch, matvec_batch};
use crate::test_support::{naive_ternary_dot, pseudo_random_values, ternary_rows};
use crate::{Q8Input, Q8Kind, dot};

const COLS: usize = 256;
const ROW_BYTES: usize = COLS / 4;
const SCALE: f32 = 1.5;

const fn ternary(data: &[u8]) -> Weights<'_> {
    Weights::Ternary {
        data,
        row_bytes: ROW_BYTES,
        scale: SCALE,
    }
}

fn run(weights: Weights<'_>, inputs: &[f32], tokens: usize) -> Vec<u32> {
    let mut out = vec![f32::NAN; weights.rows() * tokens];
    matmul_batch(inputs, tokens, &mut [(weights, &mut out[..])]);
    out.into_iter().map(f32::to_bits).collect()
}

#[test]
fn ternary_weights_count_their_rows() {
    let data = ternary_rows(6, 1);
    assert_eq!(ternary(&data).rows(), 3);
    let empty = Weights::Ternary {
        data: &data,
        row_bytes: 0,
        scale: SCALE,
    };
    assert_eq!(empty.rows(), 0);
}

#[expect(
    clippy::cast_precision_loss,
    reason = "the test dots are far below 2^24"
)]
#[test]
fn a_matvec_is_the_integer_dot_over_s_times_the_scale() {
    let data = ternary_rows(6, 2);
    let input = pseudo_random_values(COLS, 3);
    let mut quantized = Q8Input::default();
    quantized.quantize(&input, Q8Kind::Row);
    let s = quantized.scales[0];
    let expected: Vec<u32> = data
        .as_chunks::<ROW_BYTES>()
        .0
        .iter()
        .map(|row| (naive_ternary_dot(row, &quantized) as f32 / s * SCALE).to_bits())
        .collect();
    assert_eq!(run(ternary(&data), &input, 1), expected);
}

#[test]
fn a_zero_row_multiplies_to_zero() {
    let mut data = ternary_rows(4, 4);
    data[ROW_BYTES..].fill(0x55);
    let mut out = [f32::NAN; 2];
    matvec_batch(
        &pseudo_random_values(COLS, 5),
        &mut [(ternary(&data), &mut out)],
    );
    assert_ne!(out[0], 0.0);
    assert_eq!(out[1].to_bits(), 0);
}

#[test]
fn a_ternary_batch_matches_one_matvec_per_token() {
    let rows = rayon::current_num_threads() * TASKS_PER_THREAD * 2 + 3;
    let data = ternary_rows(rows * 2, 6);
    let dense: Vec<f32> = pseudo_random_values(COLS, 7);
    for tokens in [2, 5] {
        let inputs = pseudo_random_values(tokens * COLS, 8);
        let mut batch = vec![f32::NAN; rows * tokens];
        let mut dense_batch = vec![f32::NAN; tokens];
        matmul_batch(
            &inputs,
            tokens,
            &mut [
                (ternary(&data), &mut batch[..]),
                (
                    Weights::Dense {
                        data: &dense,
                        cols: COLS,
                    },
                    &mut dense_batch[..],
                ),
            ],
        );
        for (token, input) in inputs.as_chunks::<COLS>().0.iter().enumerate() {
            let single = run(ternary(&data), input, 1);
            let column: Vec<u32> = batch
                .iter()
                .skip(token)
                .step_by(tokens)
                .map(|value| value.to_bits())
                .collect();
            assert_eq!(column, single, "token {token}");
            assert_eq!(dense_batch[token].to_bits(), dot(&dense, input).to_bits());
        }
    }
}
