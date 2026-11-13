// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_core::Tensor;
use rusticide_io::Metadata;
use rusticide_model::{InMemorySource, Matrix};

use crate::config::{Activation, DecoderConfig};
use crate::rope::RopeStyle;
use crate::weights::{BlockWeights, DecoderWeights};

pub(crate) const EPS: f32 = 1e-5;
pub(crate) const TINY_VOCAB: usize = 7;

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

pub(crate) fn matrix(tensor: Tensor) -> Matrix {
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

/// `0.05 * (1 + (i + offset) % 5)` at flat index `i`.
pub(crate) fn patterned(shape: &[usize], offset: usize) -> Tensor {
    let data: Vec<f32> = (0..shape.iter().product::<usize>())
        .map(|index| 0.05 * f32::from(1 + u8::try_from((index + offset) % 5).unwrap()))
        .collect();
    tensor(shape, &data)
}

/// Embedding 4, 2 heads over 1 kv head, feed-forward 8, context 16 and a
/// [`TINY_VOCAB`]-token vocabulary.
pub(crate) const fn tiny_config(rope: RopeStyle) -> DecoderConfig {
    DecoderConfig {
        embedding_length: 4,
        block_count: 1,
        feed_forward_length: 8,
        head_count: 2,
        head_count_kv: 1,
        head_dim: 2,
        context_length: 16,
        vocab_size: TINY_VOCAB,
        rope_freq_base: 10_000.0,
        rms_norm_eps: EPS,
        rope,
        activation: Activation::Silu,
    }
}

/// A block shaped for `config`: unit norms, no biases and every matrix
/// `fill(shape, index)`, `index` counting the matrices in order.
fn shaped_block(config: &DecoderConfig, fill: impl Fn(&[usize], usize) -> Tensor) -> BlockWeights {
    let width = config.embedding_length;
    let (kv_dim, length) = (config.kv_dim().unwrap(), config.feed_forward_length);
    let matrix = |rows: usize, cols: usize, index: usize| matrix(fill(&[rows, cols], index));
    BlockWeights {
        attn_norm: vec![1.0; width],
        attn_q: matrix(width, width, 0),
        attn_k: matrix(kv_dim, width, 1),
        attn_v: matrix(kv_dim, width, 2),
        attn_q_bias: None,
        attn_k_bias: None,
        attn_v_bias: None,
        attn_output: matrix(width, width, 3),
        ffn_norm: vec![1.0; width],
        ffn_gate: matrix(length, width, 4),
        ffn_up: matrix(length, width, 5),
        ffn_down: matrix(width, length, 6),
        attn_sub_norm: None,
        ffn_sub_norm: None,
    }
}

/// A block for `config` with every matrix [`patterned`] from `offset` on.
pub(crate) fn patterned_block(config: &DecoderConfig, offset: usize) -> BlockWeights {
    shaped_block(config, |shape, index| patterned(shape, offset + index))
}

/// A block for `config` with every matrix zero.
pub(crate) fn zero_block(config: &DecoderConfig) -> BlockWeights {
    shaped_block(config, |shape, _| zeros(shape))
}

/// `blocks` behind a patterned embedding and output and a unit output norm.
pub(crate) fn tiny_weights(config: &DecoderConfig, blocks: Vec<BlockWeights>) -> DecoderWeights {
    let shape = [config.vocab_size, config.embedding_length];
    DecoderWeights {
        token_embd: matrix(patterned(&shape, 0)),
        blocks,
        output_norm: vec![1.0; config.embedding_length],
        output: Some(matrix(patterned(&shape, 11))),
    }
}

/// `row / sqrt(mean(row^2) + EPS)`, the reference rms norm without a weight.
pub(crate) fn normed(row: &[f32]) -> Vec<f32> {
    let count = f32::from(u8::try_from(row.len()).unwrap());
    let rms = (row.iter().map(|v| v * v).sum::<f32>() / count + EPS).sqrt();
    row.iter().map(|v| v / rms).collect()
}

/// Rotates adjacent pairs `(2p, 2p + 1)` by `position * base^(-2p / len)`.
pub(crate) fn rotated(row: &[f32], position: f32, base: f32) -> Vec<f32> {
    let len = f32::from(u8::try_from(row.len()).unwrap());
    let mut out = row.to_vec();
    for (pair, [low, high]) in out.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        let exponent = -2.0 * f32::from(u8::try_from(pair).unwrap()) / len;
        let (sin, cos) = (position * base.powf(exponent)).sin_cos();
        (*low, *high) = (
            low.mul_add(cos, -*high * sin),
            high.mul_add(cos, *low * sin),
        );
    }
    out
}
