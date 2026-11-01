// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use half::f16;
use rusticide_core::Tensor;
use rusticide_io::Metadata;
use rusticide_quantize::block_format;

use super::{matmul_batch, matvec_batch};
use crate::source::InMemorySource;
use crate::weights::Matrix;

/// A dense 3 x 32 matrix `d`, a `Q8_0` 2 x 32 matrix `q` and a `Q4_K`
/// 1 x 256 matrix `k`.
fn source() -> InMemorySource {
    let dense: Vec<f32> = (0..96_u8).map(|v| f32::from(v % 5) - 2.0).collect();
    let mut q8 = Vec::new();
    for scale in [0.5_f32, 0.25] {
        q8.extend(f16::from_f32(scale).to_le_bytes());
        q8.extend((0..32_u8).map(|v| (v % 9).wrapping_sub(4)));
    }
    let mut q4_k = f16::from_f32(1.0).to_le_bytes().to_vec();
    q4_k.extend(f16::from_f32(0.5).to_le_bytes());
    q4_k.extend([3_u8; 12]);
    q4_k.extend((0..128_u8).map(|v| v.wrapping_mul(37)));
    InMemorySource::new(Metadata::default())
        .with_tensor("d", Tensor::from_slice(&dense, &[3, 32]).unwrap())
        .with_quantized("q", block_format(8).unwrap(), q8, &[2, 32])
        .unwrap()
        .with_quantized("k", block_format(12).unwrap(), q4_k, &[1, 256])
        .unwrap()
}

#[test]
fn a_batch_matches_one_matvec_per_matrix() {
    let source = source();
    let dense = Matrix::load(&source, "d", 3, 32).unwrap();
    let quantized = Matrix::load(&source, "q", 2, 32).unwrap();
    let input: Vec<f32> = (0..32_u8).map(|v| f32::from(v) / 8.0 - 2.0).collect();
    let (mut d_alone, mut q_alone) = ([0.0; 3], [0.0; 2]);
    dense.matvec(&input, &mut d_alone);
    quantized.matvec(&input, &mut q_alone);
    let (mut d_batch, mut q_batch) = ([0.0; 3], [0.0; 2]);
    matvec_batch(
        &input,
        &mut [(&dense, &mut d_batch), (&quantized, &mut q_batch)],
    );
    assert_eq!((d_batch, q_batch), (d_alone, q_alone));
}

#[test]
fn a_batch_mixes_block_and_super_block_activations() {
    let mut wide_q8 = Vec::new();
    for block in 0..8_u8 {
        wide_q8.extend(f16::from_f32(0.125).to_le_bytes());
        wide_q8.extend((0..32_u8).map(|v| v.wrapping_mul(block + 3)));
    }
    let source = source()
        .with_quantized("w", block_format(8).unwrap(), wide_q8, &[1, 256])
        .unwrap();
    let block = Matrix::load(&source, "w", 1, 256).unwrap();
    let super_block = Matrix::load(&source, "k", 1, 256).unwrap();
    let input: Vec<f32> = (0..256_u16).map(|v| f32::from(v % 13) - 6.0).collect();
    let (mut w_alone, mut k_alone) = ([0.0; 1], [0.0; 1]);
    block.matvec(&input, &mut w_alone);
    super_block.matvec(&input, &mut k_alone);
    let (mut w_batch, mut k_batch) = ([0.0; 1], [0.0; 1]);
    matvec_batch(
        &input,
        &mut [(&block, &mut w_batch), (&super_block, &mut k_batch)],
    );
    assert_eq!((w_batch, k_batch), (w_alone, k_alone));
    assert!((w_alone[0] - k_alone[0]).abs() > 1.0);
}

/// `matrix x input` one token at a time, laid out by weight row.
fn per_token(matrix: &Matrix, inputs: &[f32], tokens: usize) -> Vec<u32> {
    let mut by_row = vec![0.0; matrix.rows() * tokens];
    for (token, input) in inputs.chunks_exact(matrix.cols()).enumerate() {
        let mut output = vec![0.0; matrix.rows()];
        matrix.matvec(input, &mut output);
        for (row, value) in output.into_iter().enumerate() {
            by_row[row * tokens + token] = value;
        }
    }
    by_row.into_iter().map(f32::to_bits).collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn a_token_batch_matches_one_matvec_per_token() {
    let source = source();
    let dense = Matrix::load(&source, "d", 3, 32).unwrap();
    let quantized = Matrix::load(&source, "q", 2, 32).unwrap();
    let inputs: Vec<f32> = (0..96_u8).map(|v| f32::from(v % 23) / 4.0 - 3.0).collect();
    let (mut d_batch, mut q_batch) = ([0.0; 9], [0.0; 6]);
    matmul_batch(
        &inputs,
        3,
        &mut [(&dense, &mut d_batch[..]), (&quantized, &mut q_batch[..])],
    );
    assert_eq!(bits(&d_batch), per_token(&dense, &inputs, 3));
    assert_eq!(bits(&q_batch), per_token(&quantized, &inputs, 3));
    assert_ne!(d_batch[0], d_batch[1]);
}

#[test]
fn a_token_batch_mixes_block_and_super_block_activations() {
    let mut wide_q8 = Vec::new();
    for block in 0..16_u8 {
        wide_q8.extend(f16::from_f32(0.125).to_le_bytes());
        wide_q8.extend((0..32_u8).map(|v| v.wrapping_mul(block + 3)));
    }
    let source = source()
        .with_quantized("w", block_format(8).unwrap(), wide_q8, &[2, 256])
        .unwrap();
    let block = Matrix::load(&source, "w", 2, 256).unwrap();
    let super_block = Matrix::load(&source, "k", 1, 256).unwrap();
    let inputs: Vec<f32> = (0..512_u16).map(|v| f32::from(v % 13) - 6.0).collect();
    let (mut w_batch, mut k_batch) = ([0.0; 4], [0.0; 2]);
    matmul_batch(
        &inputs,
        2,
        &mut [(&block, &mut w_batch[..]), (&super_block, &mut k_batch[..])],
    );
    assert_eq!(bits(&w_batch), per_token(&block, &inputs, 2));
    assert_eq!(bits(&k_batch), per_token(&super_block, &inputs, 2));
    assert_ne!(k_batch[0], k_batch[1]);
}

#[test]
fn zero_tokens_leave_the_outputs_untouched() {
    let dense = Matrix::load(&source(), "d", 3, 32).unwrap();
    let mut output = [7.0; 3];
    matmul_batch(&[1.0; 32], 0, &mut [(&dense, &mut output)]);
    assert_eq!(output, [7.0; 3]);
}
