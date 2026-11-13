// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_kernels::rope_half_split;
use rusticide_model::KvCache;

use super::Attention;
use crate::config::DecoderConfig;
use crate::test_support::{
    Projections, assert_close, block, config, identity, normed, picker, zeros,
};
use crate::weights::BlockWeights;

const WIDTH: usize = 8;

/// Row `token` of a deterministic embedding: `0.05 * (1 + i % 5)` by flat index.
fn row(token: usize) -> Vec<f32> {
    (token * WIDTH..(token + 1) * WIDTH)
        .map(|index| 0.05 * f32::from(1 + u8::try_from(index % 5).unwrap()))
        .collect()
}

fn added(base: &[f32], delta: &[f32]) -> Vec<f32> {
    base.iter().zip(delta).map(|(a, b)| a + b).collect()
}

fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter().zip(rhs).map(|(a, b)| a * b).sum()
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

/// An 8-wide, single-head block with identity projections.
fn wide(query_bias: Option<&[f32]>, key_bias: Option<&[f32]>) -> BlockWeights {
    block(Projections {
        query: identity(WIDTH),
        key: identity(WIDTH),
        value: identity(WIDTH),
        output: identity(WIDTH),
        biases: [
            query_bias.map(<[f32]>::to_vec),
            key_bias.map(<[f32]>::to_vec),
            None,
        ],
    })
}

/// The output of the last of `inputs`, each forwarded alone at its position.
fn run(config: &DecoderConfig, block: &BlockWeights, inputs: &[&[f32]]) -> Vec<f32> {
    let mut attention = Attention::new(config).unwrap();
    let mut cache = KvCache::new(1, config.kv_dim().unwrap(), config.context_length);
    let mut output = vec![0.0; config.embedding_length];
    for (position, input) in inputs.iter().enumerate() {
        attention
            .forward(block, &mut cache, 0, position, input, &mut output)
            .unwrap();
    }
    output
}

#[test]
fn half_split_attention_rotates_queries_and_keys() {
    let config = config(WIDTH, 1, 1);
    let (first, second) = (row(1), row(2));
    let output = run(&config, &wide(None, None), &[&first, &second]);
    let mut rotated = second.clone();
    rope_half_split(&mut rotated, 1, config.rope_freq_base, WIDTH / 2);
    let mixed = attend_wide(&rotated, [&first, &rotated], [&first, &second]);
    assert_close(&output, &mixed);
}

#[test]
fn attention_adds_the_value_bias() {
    let config = config(4, 2, 1);
    let block = block(Projections {
        query: identity(4),
        key: picker(2, 4, &[(0, 0), (1, 1)]),
        value: zeros(&[2, 4]),
        output: identity(4),
        biases: [
            Some(vec![0.1, 0.2, 0.3, 0.4]),
            Some(vec![0.5, 0.6]),
            Some(vec![0.5, -0.25]),
        ],
    });
    let output = run(&config, &block, &[&[1.0, 2.0, 3.0, 4.0]]);
    assert_close(&output, &[0.5, -0.25, 0.5, -0.25]);
}

#[test]
fn attention_adds_query_and_key_biases_before_rope() {
    let query_bias = [0.5, -1.0, 0.25, 2.0, -0.5, 1.5, 0.75, -0.25];
    let key_bias = [1.0, 0.5, -0.5, 0.25, -1.5, 0.75, 2.0, -1.0];
    let config = config(WIDTH, 1, 1);
    let (first, second) = (row(1), row(2));
    let block = wide(Some(&query_bias), Some(&key_bias));
    let output = run(&config, &block, &[&first, &second]);
    let base = config.rope_freq_base;
    let rotate = |vector: &[f32]| {
        let mut out = vector.to_vec();
        rope_half_split(&mut out, 1, base, WIDTH / 2);
        out
    };
    let query = rotate(&added(&second, &query_bias));
    let first_key = added(&first, &key_bias);
    let second_key = rotate(&added(&second, &key_bias));
    let expected = attend_wide(&query, [&first_key, &second_key], [&first, &second]);
    assert_close(&output, &expected);
    let late_query = added(&rotate(&second), &query_bias);
    let late_key = added(&rotate(&second), &key_bias);
    let late = attend_wide(&late_query, [&first_key, &late_key], [&first, &second]);
    assert!(output.iter().zip(&late).any(|(a, b)| (a - b).abs() > 1e-3));
}

#[test]
fn batched_rotation_matches_one_token_at_a_time() {
    let bias = [0.5, -1.0, 0.25, 2.0, -0.5, 1.5, 0.75, -0.25];
    let config = config(WIDTH, 1, 1);
    let block = wide(Some(&bias), Some(&bias));
    let tokens = 12;
    let input: Vec<f32> = (0..tokens * WIDTH)
        .map(|i| (f32::from(u8::try_from(i % 13).unwrap()) - 4.0) * 0.25)
        .collect();
    let forward = |batches: &[&[f32]]| {
        let mut attention = Attention::new(&config).unwrap();
        let mut cache = KvCache::new(1, WIDTH, config.context_length);
        let (mut output, mut start) = (Vec::new(), 0);
        for batch in batches {
            let mut out = vec![0.0; batch.len()];
            attention
                .forward(&block, &mut cache, 0, start, batch, &mut out)
                .unwrap();
            start += batch.len() / WIDTH;
            output.extend(out);
        }
        (output, cache.keys(0).to_vec(), cache.values(0).to_vec())
    };
    let singles: Vec<&[f32]> = input
        .as_chunks::<WIDTH>()
        .0
        .iter()
        .map(<[f32; WIDTH]>::as_slice)
        .collect();
    let (batch_out, batch_keys, batch_values) = forward(&[&input]);
    let (single_out, single_keys, single_values) = forward(&singles);
    assert_eq!(batch_keys, single_keys);
    assert_eq!(batch_values, single_values);
    assert_eq!(
        batch_out[(tokens - 1) * WIDTH..],
        single_out[(tokens - 1) * WIDTH..]
    );
}

#[test]
fn attention_rejects_an_invalid_config_instead_of_panicking() {
    let mut config = config(WIDTH, 2, 1);
    config.head_count_kv = 0;
    match Attention::new(&config) {
        Err(rusticide_model::ModelError::InvalidConfig { reason }) => {
            assert_eq!(
                reason,
                "kv head count must be non-zero and divide the head count"
            );
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn attn_sub_norm_normalizes_each_token_over_all_heads_before_the_output() {
    let config = config(4, 2, 2);
    let weight = vec![1.0, 2.0, 3.0, 4.0];
    let mut block = block(Projections {
        query: zeros(&[4, 4]),
        key: identity(4),
        value: identity(4),
        output: identity(4),
        biases: [None, None, None],
    });
    block.attn_sub_norm = Some(weight.clone());
    let (first, second) = ([1.0, 2.0, 3.0, 4.0], [5.0, -1.0, 0.5, 2.0]);
    let mut attention = Attention::new(&config).unwrap();
    let mut cache = KvCache::new(1, 4, config.context_length);
    let input: Vec<f32> = first.iter().chain(&second).copied().collect();
    let mut output = vec![0.0; input.len()];
    attention
        .forward(&block, &mut cache, 0, 0, &input, &mut output)
        .unwrap();
    let mean: Vec<f32> = first
        .iter()
        .zip(&second)
        .map(|(a, b)| f32::midpoint(*a, *b))
        .collect();
    let expected: Vec<f32> = [first.to_vec(), mean]
        .iter()
        .flat_map(|mixed| {
            let norm = normed(mixed);
            norm.iter()
                .zip(&weight)
                .map(|(n, w)| n * w)
                .collect::<Vec<_>>()
        })
        .collect();
    assert_close(&output, &expected);
}
