// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_core::Tensor;
use rusticide_io::GGUFValue;
use rusticide_kernels::rope_half_split;
use rusticide_model::{InMemorySource, KvCache, Model};

use crate::attention::Attention;
use crate::config::LlamaConfig;
use crate::test_support::{assert_close, identity, normed, picker, rotated, tensor, zeros};
use crate::testing::{TinySource, tiny_tensor};
use crate::weights::LlamaWeights;
use crate::{Llama, Variant};

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
    Llama::from_source(source, Variant::Llama).unwrap()
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

#[test]
fn qwen2_attention_uses_half_split_rope() {
    let source = wide(1, 1, WIDTH)
        .with_tensor("blk.0.attn_q.weight", identity(WIDTH))
        .with_tensor("blk.0.attn_k.weight", identity(WIDTH))
        .with_tensor("blk.0.attn_v.weight", identity(WIDTH))
        .build();
    let mut config = LlamaConfig::from_source(&source, Variant::Llama).unwrap();
    let weights = LlamaWeights::load(&source, &config).unwrap();
    config.variant = Variant::Qwen2;
    let mut attention = Attention::new(&config).unwrap();
    let mut cache = KvCache::new(1, WIDTH, config.context_length);
    let (first, second) = (row(1), row(2));
    let mut output = vec![0.0; WIDTH];
    attention
        .forward(&weights.blocks[0], &mut cache, 0, 0, &first, &mut output)
        .unwrap();
    attention
        .forward(&weights.blocks[0], &mut cache, 0, 1, &second, &mut output)
        .unwrap();
    let mut rotated = second.clone();
    rope_half_split(&mut rotated, 1, config.rope_freq_base, WIDTH / 2);
    let mixed = attend_wide(&rotated, [&first, &rotated], [&first, &second]);
    assert_close(&output, &mixed);
}

fn attend_wide(query: &[f32], keys: [&[f32]; 2], values: [&[f32]; 2]) -> Vec<f32> {
    let scores = keys.map(|key| dot(query, key) / 8.0_f32.sqrt());
    let top = scores[0].max(scores[1]);
    let weights = scores.map(|score| (score - top).exp());
    let total = weights[0] + weights[1];
    (0..WIDTH)
        .map(|index| weights[1].mul_add(values[1][index], weights[0] * values[0][index]) / total)
        .collect()
}

fn qwen2_outputs(source: &InMemorySource, inputs: &[[f32; 4]]) -> Vec<f32> {
    let config = LlamaConfig::from_source(source, Variant::Qwen2).unwrap();
    let weights = LlamaWeights::load(source, &config).unwrap();
    let mut attention = Attention::new(&config).unwrap();
    let mut cache = KvCache::new(1, config.kv_dim().unwrap(), config.context_length);
    let mut output = vec![0.0; 4];
    for (position, input) in inputs.iter().enumerate() {
        attention
            .forward(
                &weights.blocks[0],
                &mut cache,
                0,
                position,
                input,
                &mut output,
            )
            .unwrap();
    }
    output
}

#[test]
fn qwen2_adds_the_value_bias() {
    let source = TinySource::qwen2()
        .with_tensor("blk.0.attn_v.weight", zeros(&[2, 4]))
        .with_tensor("blk.0.attn_v.bias", tensor(&[2], &[0.5, -0.25]))
        .with_tensor("blk.0.attn_output.weight", identity(4))
        .build();
    let output = qwen2_outputs(&source, &[[1.0, 2.0, 3.0, 4.0]]);
    assert_close(&output, &[0.5, -0.25, 0.5, -0.25]);
}

#[test]
fn qwen2_query_and_key_biases_change_the_scores() {
    let inputs = [[1.0, 2.0, 3.0, 4.0], [-2.0, 0.5, 1.0, -1.0]];
    let biased = qwen2_outputs(&TinySource::qwen2().build(), &inputs);
    let plain = TinySource::qwen2()
        .with_tensor("blk.0.attn_q.bias", zeros(&[4]))
        .build();
    let without_query = qwen2_outputs(&plain, &inputs);
    let without_key = qwen2_outputs(
        &TinySource::qwen2()
            .with_tensor("blk.0.attn_k.bias", zeros(&[2]))
            .build(),
        &inputs,
    );
    let differs =
        |left: &[f32], right: &[f32]| left.iter().zip(right).any(|(a, b)| (a - b).abs() > 1e-4);
    assert!(differs(&biased, &without_query));
    assert!(differs(&biased, &without_key));
}
