// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_core::Tensor;
use rusticide_io::GGUFValue;
use rusticide_model::{InMemorySource, Model};

use crate::Llama;
use crate::test_support::{assert_close, identity, normed, picker, rotated, tensor, zeros};
use crate::testing::{TinySource, tiny_tensor};

const WIDTH: usize = 8;

fn embedding() -> Vec<f32> {
    tiny_tensor("token_embd.weight", &[WIDTH, WIDTH], 0)
        .unwrap()
        .to_vec::<f32>()
        .unwrap()
}

fn row(token: usize) -> Vec<f32> {
    embedding()[token * WIDTH..(token + 1) * WIDTH].to_vec()
}

/// An 8-wide, 8-token model with every block weight zero and an identity output.
fn wide(heads: u32, kv_heads: u32, kv_dim: usize) -> TinySource {
    let ones = || tensor(&[WIDTH], &[1.0; WIDTH]);
    let embedding: Tensor = tensor(&[WIDTH, WIDTH], &embedding());
    TinySource::new()
        .with_metadata("llama.embedding_length", GGUFValue::Uint32(8))
        .with_metadata("llama.attention.head_count", GGUFValue::Uint32(heads))
        .with_metadata("llama.attention.head_count_kv", GGUFValue::Uint32(kv_heads))
        .with_tensor("token_embd.weight", embedding)
        .with_tensor("output.weight", identity(WIDTH))
        .with_tensor("output_norm.weight", ones())
        .with_tensor("blk.0.attn_norm.weight", ones())
        .with_tensor("blk.0.ffn_norm.weight", ones())
        .with_tensor("blk.0.attn_q.weight", zeros(&[WIDTH, WIDTH]))
        .with_tensor("blk.0.attn_k.weight", zeros(&[kv_dim, WIDTH]))
        .with_tensor("blk.0.attn_v.weight", zeros(&[kv_dim, WIDTH]))
        .with_tensor("blk.0.attn_output.weight", identity(WIDTH))
        .with_tensor("blk.0.ffn_gate.weight", zeros(&[8, WIDTH]))
        .with_tensor("blk.0.ffn_up.weight", zeros(&[8, WIDTH]))
        .with_tensor("blk.0.ffn_down.weight", zeros(&[WIDTH, 8]))
}

fn model(source: &InMemorySource) -> Llama {
    Llama::from_source(source).unwrap()
}

fn added(base: &[f32], delta: &[f32]) -> Vec<f32> {
    base.iter().zip(delta).map(|(a, b)| a + b).collect()
}

#[test]
fn query_head_reads_the_kv_head_of_its_group() {
    let picks = [(0, 0), (1, 1), (2, 2), (3, 3)];
    let source = wide(4, 2, 4)
        .with_tensor("blk.0.attn_v.weight", picker(4, WIDTH, &picks))
        .build();
    let logits = model(&source).forward(3, 0).unwrap();
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
    let source = wide(2, 1, 4)
        .with_metadata("llama.rope.freq_base", GGUFValue::Float32(100.0))
        .with_tensor("blk.0.attn_q.weight", identity(WIDTH))
        .with_tensor("blk.0.attn_k.weight", picker(4, WIDTH, &keys))
        .with_tensor("blk.0.attn_v.weight", picker(4, WIDTH, &values))
        .build();
    let mut llama = model(&source);
    llama.forward(1, 0).unwrap();
    let logits = llama.forward(2, 1).unwrap();
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
