// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use half::f16;

use super::{Activations, Weights, matmul_batch, matvec_batch};
use crate::{Q8Input, Q8Kind, QuantKernel, dot};

const Q8_0: u32 = 8;
const Q4_K: u32 = 12;

/// The row-major kernel: these matrices are not prepared for an adapter.
fn kernel(type_id: u32) -> &'static QuantKernel {
    crate::portable::ADAPTER.quant_kernel(type_id).unwrap()
}

/// A row-major `Q8_0` kernel paired with per-row activations.
static ROW_Q8_0: QuantKernel = QuantKernel {
    activation: Q8Kind::Row,
    ..crate::portable::Q8_0
};

/// A dense 3 x 32 matrix.
fn dense_data() -> Vec<f32> {
    (0..96_u8).map(|v| f32::from(v % 5) - 2.0).collect()
}

/// A `Q8_0` matrix of one 32-wide block per scale.
fn q8_data(scales: &[f32]) -> Vec<u8> {
    let mut data = Vec::new();
    for (block, scale) in (0..=u8::MAX).cycle().zip(scales) {
        data.extend(f16::from_f32(*scale).to_le_bytes());
        data.extend((0..32_u8).map(|v| v.wrapping_mul(block.wrapping_add(3))));
    }
    data
}

/// A `Q4_K` 1 x 256 matrix.
fn q4_k_data() -> Vec<u8> {
    let mut data = f16::from_f32(1.0).to_le_bytes().to_vec();
    data.extend(f16::from_f32(0.5).to_le_bytes());
    data.extend([3_u8; 12]);
    data.extend((0..128_u8).map(|v| v.wrapping_mul(37)));
    data
}

fn blocks(type_id: u32, data: &[u8], row_bytes: usize) -> Weights<'_> {
    Weights::Blocks {
        kernel: kernel(type_id),
        data,
        row_bytes,
    }
}

/// `weights x input` computed row by row with the bare kernels.
fn reference(weights: &Weights<'_>, input: &[f32]) -> Vec<f32> {
    match weights {
        Weights::Dense { data, cols } => data
            .chunks_exact(*cols)
            .map(|row| dot(row, input))
            .collect(),
        Weights::Blocks {
            kernel,
            data,
            row_bytes,
        } => {
            let mut quantized = Q8Input::default();
            quantized.quantize(input, kernel.activation());
            let rows = data.chunks_exact(*row_bytes);
            rows.map(|row| kernel.dot_row(row, &quantized)).collect()
        }
        Weights::Ternary { .. } => unreachable!("see matmul_ternary_tests"),
    }
}

/// `weights x input` one token at a time, laid out by weight row.
fn per_token(weights: Weights<'_>, inputs: &[f32], cols: usize) -> Vec<u32> {
    let tokens = inputs.len() / cols;
    let mut by_row = vec![0.0; weights.rows() * tokens];
    for (token, input) in inputs.chunks_exact(cols).enumerate() {
        let mut output = vec![0.0; weights.rows()];
        matvec_batch(input, &mut [(weights, &mut output)]);
        for (row, value) in output.into_iter().enumerate() {
            by_row[row * tokens + token] = value;
        }
    }
    bits(&by_row)
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn weights_count_their_rows() {
    let (dense, q8) = (dense_data(), q8_data(&[0.5, 0.25]));
    assert_eq!(
        Weights::Dense {
            data: &dense,
            cols: 32
        }
        .rows(),
        3
    );
    assert_eq!(blocks(Q8_0, &q8, 34).rows(), 2);
    assert_eq!(
        Weights::Dense {
            data: &dense,
            cols: 0
        }
        .rows(),
        0
    );
}

#[test]
fn a_batch_matches_one_matvec_per_matrix() {
    let (dense, q8) = (dense_data(), q8_data(&[0.5, 0.25]));
    let dense = Weights::Dense {
        data: &dense,
        cols: 32,
    };
    let quantized = blocks(Q8_0, &q8, 34);
    let input: Vec<f32> = (0..32_u8).map(|v| f32::from(v) / 8.0 - 2.0).collect();
    let (mut d_batch, mut q_batch) = ([0.0; 3], [0.0; 2]);
    matvec_batch(
        &input,
        &mut [(dense, &mut d_batch), (quantized, &mut q_batch)],
    );
    assert_eq!(bits(&d_batch), bits(&reference(&dense, &input)));
    assert_eq!(bits(&q_batch), bits(&reference(&quantized, &input)));
}

#[test]
fn a_batch_mixes_block_and_super_block_activations() {
    let (wide, q4_k) = (q8_data(&[0.125; 8]), q4_k_data());
    let block = blocks(Q8_0, &wide, 272);
    let super_block = blocks(Q4_K, &q4_k, 144);
    let input: Vec<f32> = (0..256_u16).map(|v| f32::from(v % 13) - 6.0).collect();
    let (mut w_batch, mut k_batch) = ([0.0; 1], [0.0; 1]);
    matvec_batch(
        &input,
        &mut [(block, &mut w_batch), (super_block, &mut k_batch)],
    );
    assert_eq!(bits(&w_batch), bits(&reference(&block, &input)));
    assert_eq!(bits(&k_batch), bits(&reference(&super_block, &input)));
    assert!((w_batch[0] - k_batch[0]).abs() > 1.0);
}

#[test]
fn a_token_batch_matches_one_matvec_per_token() {
    let (dense, q8) = (dense_data(), q8_data(&[0.5, 0.25]));
    let dense = Weights::Dense {
        data: &dense,
        cols: 32,
    };
    let quantized = blocks(Q8_0, &q8, 34);
    let inputs: Vec<f32> = (0..96_u8).map(|v| f32::from(v % 23) / 4.0 - 3.0).collect();
    let (mut d_batch, mut q_batch) = ([0.0; 9], [0.0; 6]);
    matmul_batch(
        &inputs,
        3,
        &mut [(dense, &mut d_batch[..]), (quantized, &mut q_batch[..])],
    );
    assert_eq!(bits(&d_batch), per_token(dense, &inputs, 32));
    assert_eq!(bits(&q_batch), per_token(quantized, &inputs, 32));
    assert_ne!(d_batch[0], d_batch[1]);
}

#[test]
fn a_token_batch_mixes_block_and_super_block_activations() {
    let (wide, q4_k) = (q8_data(&[0.125; 16]), q4_k_data());
    let block = blocks(Q8_0, &wide, 272);
    let super_block = blocks(Q4_K, &q4_k, 144);
    let inputs: Vec<f32> = (0..512_u16).map(|v| f32::from(v % 13) - 6.0).collect();
    let (mut w_batch, mut k_batch) = ([0.0; 4], [0.0; 2]);
    matmul_batch(
        &inputs,
        2,
        &mut [(block, &mut w_batch[..]), (super_block, &mut k_batch[..])],
    );
    assert_eq!(bits(&w_batch), per_token(block, &inputs, 256));
    assert_eq!(bits(&k_batch), per_token(super_block, &inputs, 256));
    assert_ne!(k_batch[0], k_batch[1]);
}

#[test]
fn zero_tokens_leave_the_outputs_untouched() {
    let dense = dense_data();
    let mut output = [7.0; 3];
    let weights = Weights::Dense {
        data: &dense,
        cols: 32,
    };
    matmul_batch(&[1.0; 32], 0, &mut [(weights, &mut output)]);
    assert_eq!(output, [7.0; 3]);
}

#[test]
fn a_job_with_no_rows_does_no_work() {
    let dense = dense_data();
    let weights = Weights::Dense {
        data: &dense,
        cols: 32,
    };
    let no_width = Weights::Dense {
        data: &dense,
        cols: 0,
    };
    let no_data = blocks(Q8_0, &[], 0);
    let (mut output, mut first, mut second) = ([0.0; 3], [7.0; 2], [7.0; 2]);
    let jobs = &mut [
        (no_width, &mut first[..]),
        (weights, &mut output[..]),
        (no_data, &mut second[..]),
    ];
    matvec_batch(&[1.0; 32], jobs);
    assert_eq!((first, second), ([7.0; 2], [7.0; 2]));
    assert_eq!(bits(&output), bits(&reference(&weights, &[1.0; 32])));
}

#[test]
fn more_tokens_than_input_values_leave_the_outputs_untouched() {
    let dense = dense_data();
    let weights = Weights::Dense {
        data: &dense,
        cols: 32,
    };
    let mut output = [7.0; 3];
    matmul_batch(&[1.0; 32], usize::MAX, &mut [(weights, &mut output)]);
    matmul_batch(&[1.0; 32], 33, &mut [(weights, &mut output)]);
    matmul_batch(&[], 1, &mut [(weights, &mut output)]);
    assert_eq!(output, [7.0; 3]);
}

#[test]
fn tasks_spanning_several_rows_match_a_per_row_reference() {
    let rows = rayon::current_num_threads() * super::TASKS_PER_THREAD * 2 + 3;
    let tokens = 2;
    let dense: Vec<f32> = [-3.0, -1.0, 0.0, 2.0, 5.0, 1.0, -2.0]
        .into_iter()
        .cycle()
        .take(rows * 32)
        .collect();
    let scales: Vec<f32> = [0.125, 0.25, 0.375, 0.5, 0.625]
        .into_iter()
        .cycle()
        .take(rows)
        .collect();
    let q8 = q8_data(&scales);
    let dense = Weights::Dense {
        data: &dense,
        cols: 32,
    };
    let quantized = blocks(Q8_0, &q8, 34);
    let inputs: Vec<f32> = (0..64_u8).map(|v| f32::from(v % 11) / 4.0 - 1.0).collect();
    let (mut d_batch, mut q_batch) = (vec![0.0; rows * tokens], vec![0.0; rows * tokens]);
    matmul_batch(
        &inputs,
        tokens,
        &mut [(dense, &mut d_batch[..]), (quantized, &mut q_batch[..])],
    );
    for (weights, batch) in [(dense, &d_batch), (quantized, &q_batch)] {
        for (token, input) in inputs.as_chunks::<32>().0.iter().enumerate() {
            let expected = reference(&weights, input);
            let got: Vec<f32> = batch.iter().skip(token).step_by(tokens).copied().collect();
            assert_eq!(bits(&got), bits(&expected));
        }
    }
}

#[test]
fn batched_quantization_matches_per_token_quantization() {
    const WIDTH: usize = 256;
    for tokens in [1_usize, 2, 7, 64] {
        let inputs: Vec<f32> = (0..tokens * WIDTH)
            .map(|v| f32::from(u8::try_from(v * 7 % 23).unwrap()) / 5.0 - 2.0)
            .collect();
        for kernel in [kernel(Q8_0), kernel(Q4_K), &ROW_Q8_0] {
            let kind = kernel.activation();
            let weights = Weights::Blocks {
                kernel,
                data: &[],
                row_bytes: 1,
            };
            let activations = Activations::new(&inputs, tokens, [&weights].into_iter());
            let batched = activations.quantized(kind);
            assert_eq!(batched.len(), tokens);
            for (got, row) in batched.iter().zip(inputs.as_chunks::<WIDTH>().0) {
                let mut expected = Q8Input::default();
                expected.quantize(row, kind);
                assert_eq!(bits(&got.scales), bits(&expected.scales));
                assert_eq!(got.quants, expected.quants);
                assert_eq!(got.sums, expected.sums);
            }
        }
    }
}
