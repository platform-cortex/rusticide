// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_core::Tensor;
use rusticide_io::Metadata;
use rusticide_model::{InMemorySource, Matrix};

use crate::config::{Activation, DecoderConfig};
use crate::rope::RopeStyle;
use crate::weights::BlockWeights;

pub(crate) fn tensor(shape: &[usize], data: &[f32]) -> Tensor {
    Tensor::from_slice(data, shape).unwrap()
}

pub(crate) fn zeros(shape: &[usize]) -> Tensor {
    tensor(shape, &vec![0.0; shape.iter().product()])
}

/// A `rows x cols` matrix holding `1.0` at each `(row, col)` and `0.0` elsewhere.
pub(crate) fn picker(rows: usize, cols: usize, picks: &[(usize, usize)]) -> Tensor {
    let mut data = vec![0.0; rows * cols];
    for &(row, col) in picks {
        data[row * cols + col] = 1.0;
    }
    tensor(&[rows, cols], &data)
}

pub(crate) fn identity(size: usize) -> Tensor {
    let picks: Vec<(usize, usize)> = (0..size).map(|index| (index, index)).collect();
    picker(size, size, &picks)
}

fn matrix(tensor: Tensor) -> Matrix {
    let dims = tensor.shape().dims().to_vec();
    let source = InMemorySource::new(Metadata::default()).with_tensor("m", tensor);
    Matrix::load(&source, "m", dims[0], dims[1]).unwrap()
}

/// A `width`-wide config with `heads` query heads over `kv_heads` kv heads.
pub(crate) fn config(width: usize, heads: usize, kv_heads: usize) -> DecoderConfig {
    DecoderConfig {
        embedding_length: width,
        block_count: 1,
        feed_forward_length: width,
        head_count: heads,
        head_count_kv: kv_heads,
        head_dim: width / heads,
        context_length: 16,
        vocab_size: width,
        rope_freq_base: 10_000.0,
        rms_norm_eps: 1e-5,
        rope: RopeStyle::HalfSplit,
        activation: Activation::Silu,
    }
}

/// The attention projections of a block and their optional biases.
pub(crate) struct Projections {
    pub(crate) query: Tensor,
    pub(crate) key: Tensor,
    pub(crate) value: Tensor,
    pub(crate) output: Tensor,
    pub(crate) biases: [Option<Vec<f32>>; 3],
}

/// A block with the given attention and a zero feed-forward.
pub(crate) fn block(projections: Projections) -> BlockWeights {
    let width = projections.output.shape().dims()[0];
    let [query_bias, key_bias, value_bias] = projections.biases;
    BlockWeights {
        attn_norm: vec![1.0; width],
        attn_q: matrix(projections.query),
        attn_k: matrix(projections.key),
        attn_v: matrix(projections.value),
        attn_q_bias: query_bias,
        attn_k_bias: key_bias,
        attn_v_bias: value_bias,
        attn_output: matrix(projections.output),
        ffn_norm: vec![1.0; width],
        ffn_gate: matrix(zeros(&[width, width])),
        ffn_up: matrix(zeros(&[width, width])),
        ffn_down: matrix(zeros(&[width, width])),
        attn_sub_norm: None,
        ffn_sub_norm: None,
    }
}

pub(crate) fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < 1e-5, "{actual:?} vs {expected:?}");
    }
}
