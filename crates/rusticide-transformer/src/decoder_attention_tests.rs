// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_core::Tensor;
use rusticide_model::Model;

use super::Decoder;
use crate::config::{Activation, DecoderConfig};
use crate::rope::RopeStyle;
use crate::test_support::{
    EPS, assert_close, identity, matrix, normed, patterned, picker, rotated, zero_block,
};
use crate::weights::{BlockWeights, DecoderWeights};

const WIDTH: usize = 8;

fn embedding() -> Vec<f32> {
    patterned(&[WIDTH, WIDTH], 0).to_vec::<f32>().unwrap()
}

fn row(token: usize) -> Vec<f32> {
    embedding()[token * WIDTH..(token + 1) * WIDTH].to_vec()
}

/// An 8-wide, 8-token config with `heads` query heads over `kv_heads`.
const fn wide_config(heads: usize, kv_heads: usize, rope_freq_base: f32) -> DecoderConfig {
    DecoderConfig {
        embedding_length: WIDTH,
        block_count: 1,
        feed_forward_length: WIDTH,
        head_count: heads,
        head_count_kv: kv_heads,
        head_dim: WIDTH / heads,
        context_length: 16,
        vocab_size: WIDTH,
        rope_freq_base,
        rms_norm_eps: EPS,
        rope: RopeStyle::Adjacent,
        activation: Activation::Silu,
    }
}

/// A decoder over `config` with every block weight zero but those `edit`
/// sets, an identity `attn_output` and an identity output.
fn wide(config: DecoderConfig, edit: impl FnOnce(&mut BlockWeights)) -> Decoder {
    let mut block = zero_block(&config);
    block.attn_output = matrix(identity(WIDTH));
    edit(&mut block);
    let weights = DecoderWeights {
        token_embd: matrix(Tensor::from_slice(&embedding(), &[WIDTH, WIDTH]).unwrap()),
        blocks: vec![block],
        output_norm: vec![1.0; WIDTH],
        output: Some(matrix(identity(WIDTH))),
    };
    Decoder::new(config, weights).unwrap()
}

fn added(base: &[f32], delta: &[f32]) -> Vec<f32> {
    base.iter().zip(delta).map(|(a, b)| a + b).collect()
}

#[test]
fn query_head_reads_the_kv_head_of_its_group() {
    let picks = [(0, 0), (1, 1), (2, 2), (3, 3)];
    let mut decoder = wide(wide_config(4, 2, 10_000.0), |block| {
        block.attn_v = matrix(picker(4, WIDTH, &picks));
    });
    let logits = decoder.forward(3, 0).unwrap();
    let value = normed(&row(3));
    let mixed = [
        value[0], value[1], value[0], value[1], value[2], value[3], value[2], value[3],
    ];
    assert_close(&logits, &normed(&added(&row(3), &mixed)));
}

fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter().zip(rhs).map(|(a, b)| a * b).sum()
}

fn attend(query: &[f32], keys: [&[f32]; 2], values: [&[f32]; 2]) -> Vec<f32> {
    let scores = keys.map(|key| dot(query, key) / 2.0);
    let top = scores[0].max(scores[1]);
    let weights = scores.map(|score| (score - top).exp());
    let total = weights[0] + weights[1];
    (0..4)
        .map(|index| weights[1].mul_add(values[1][index], weights[0] * values[0][index]) / total)
        .collect()
}

#[test]
fn attention_uses_queries_and_keys_rotated_per_head() {
    let keys = [(0, 0), (1, 1), (2, 2), (3, 3)];
    let values = [(0, 4), (1, 5), (2, 6), (3, 7)];
    let mut decoder = wide(wide_config(2, 1, 100.0), |block| {
        block.attn_q = matrix(identity(WIDTH));
        block.attn_k = matrix(picker(4, WIDTH, &keys));
        block.attn_v = matrix(picker(4, WIDTH, &values));
    });
    decoder.forward(1, 0).unwrap();
    let logits = decoder.forward(2, 1).unwrap();
    let (first, second) = (normed(&row(1)), normed(&row(2)));
    let cached_keys = [first[..4].to_vec(), rotated(&second[..4], 1.0, 100.0)];
    let key_refs = [cached_keys[0].as_slice(), cached_keys[1].as_slice()];
    let value_refs = [&first[4..], &second[4..]];
    let mut mixed = attend(&rotated(&second[..4], 1.0, 100.0), key_refs, value_refs);
    mixed.extend(attend(
        &rotated(&second[4..], 1.0, 100.0),
        key_refs,
        value_refs,
    ));
    assert_close(&logits, &normed(&added(&row(2), &mixed)));
}
