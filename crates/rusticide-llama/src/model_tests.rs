// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_core::Tensor;
use rusticide_io::GGUFValue;
use rusticide_model::{InMemorySource, Model};

use super::Llama;
use crate::test_support::{assert_close, identity, normed, picker, tensor, zeros};
use crate::testing::TinySource;

const EMBED: [[f32; 4]; 7] = [
    [1.0, 2.0, 3.0, 4.0],
    [2.0, -1.0, 0.5, 1.0],
    [0.5, 0.5, -2.0, 1.5],
    [-1.0, 3.0, 1.0, 0.0],
    [1.5, -0.5, 2.0, -1.0],
    [0.0, 1.0, 0.0, 2.0],
    [3.0, 1.0, -1.0, 0.5],
];
const OUTPUT: [[f32; 4]; 7] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [1.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 2.0, 0.0],
];
const OUTPUT_NORM: [f32; 4] = [1.0, 0.5, 2.0, 1.0];

fn rows(values: &[[f32; 4]; 7]) -> Tensor {
    tensor(&[7, 4], values.as_flattened())
}

fn zero_block(mut source: TinySource, block: usize) -> TinySource {
    let shapes: [(&str, [usize; 2]); 7] = [
        ("attn_q", [4, 4]),
        ("attn_k", [2, 4]),
        ("attn_v", [2, 4]),
        ("attn_output", [4, 4]),
        ("ffn_gate", [8, 4]),
        ("ffn_up", [8, 4]),
        ("ffn_down", [4, 8]),
    ];
    for (part, shape) in shapes {
        source = source.with_tensor(&format!("blk.{block}.{part}.weight"), zeros(&shape));
    }
    source
        .with_tensor(
            &format!("blk.{block}.attn_norm.weight"),
            tensor(&[4], &[1.0; 4]),
        )
        .with_tensor(
            &format!("blk.{block}.ffn_norm.weight"),
            tensor(&[4], &[1.0; 4]),
        )
}

/// The tiny model with literal embedding and output and every block weight zero.
fn quiet() -> TinySource {
    let source = TinySource::new()
        .with_tensor("token_embd.weight", rows(&EMBED))
        .with_tensor("output.weight", rows(&OUTPUT))
        .with_tensor("output_norm.weight", tensor(&[4], &OUTPUT_NORM));
    zero_block(source, 0)
}

fn model(source: &InMemorySource) -> Llama {
    Llama::from_source(source).unwrap()
}

fn logits_for(hidden: &[f32], output: &[[f32; 4]; 7]) -> Vec<f32> {
    let scaled: Vec<f32> = normed(hidden)
        .iter()
        .zip(OUTPUT_NORM)
        .map(|(value, weight)| value * weight)
        .collect();
    output
        .iter()
        .map(|row| row.iter().zip(&scaled).map(|(a, b)| a * b).sum())
        .collect()
}

fn plus(base: &[f32; 4], delta: &[f32; 4]) -> [f32; 4] {
    std::array::from_fn(|index| base[index] + delta[index])
}

#[test]
fn zero_blocks_leave_the_normed_embedding() {
    let logits = model(&quiet().build()).forward(0, 0).unwrap();
    assert_close(&logits, &logits_for(&EMBED[0], &OUTPUT));
}

#[test]
fn missing_output_weight_ties_to_the_embedding() {
    let source = quiet().without_tensor("output.weight").build();
    let logits = model(&source).forward(3, 0).unwrap();
    assert_close(&logits, &logits_for(&EMBED[3], &EMBED));
}

#[test]
fn second_position_averages_values_when_queries_are_zero() {
    let source = quiet()
        .with_tensor("blk.0.attn_k.weight", picker(2, 4, &[(0, 2), (1, 3)]))
        .with_tensor("blk.0.attn_v.weight", picker(2, 4, &[(0, 0), (1, 1)]))
        .with_tensor("blk.0.attn_output.weight", identity(4))
        .build();
    let mut llama = model(&source);
    llama.forward(1, 0).unwrap();
    let logits = llama.forward(4, 1).unwrap();
    let (first, second) = (normed(&EMBED[1]), normed(&EMBED[4]));
    let mean = [
        f32::midpoint(first[0], second[0]),
        f32::midpoint(first[1], second[1]),
    ];
    let hidden = plus(&EMBED[4], &[mean[0], mean[1], mean[0], mean[1]]);
    assert_close(&logits, &logits_for(&hidden, &OUTPUT));
}

#[test]
fn grouped_query_heads_share_their_kv_head() {
    let attn_norm = [1.0, 3.0, 1.0, 1.0];
    let source = quiet()
        .with_tensor("blk.0.attn_norm.weight", tensor(&[4], &attn_norm))
        .with_tensor("blk.0.attn_q.weight", identity(4))
        .with_tensor("blk.0.attn_v.weight", picker(2, 4, &[(0, 0), (1, 1)]))
        .with_tensor("blk.0.attn_output.weight", identity(4))
        .build();
    let logits = model(&source).forward(5, 0).unwrap();
    let normed = normed(&EMBED[5]);
    let value = [normed[0] * attn_norm[0], normed[1] * attn_norm[1]];
    let hidden = plus(&EMBED[5], &[value[0], value[1], value[0], value[1]]);
    assert_close(&logits, &logits_for(&hidden, &OUTPUT));
}

fn gated(source: TinySource, block: usize) -> TinySource {
    let name = |part: &str| format!("blk.{block}.{part}.weight");
    source
        .with_tensor(&name("ffn_norm"), tensor(&[4], &[2.0, 1.0, 1.0, 1.0]))
        .with_tensor(&name("ffn_gate"), picker(8, 4, &[(0, 0)]))
        .with_tensor(&name("ffn_up"), picker(8, 4, &[(0, 1)]))
        .with_tensor(&name("ffn_down"), picker(4, 8, &[(2, 0)]))
}

fn gated_expectation(token: usize) -> Vec<f32> {
    let normed = normed(&EMBED[token]);
    let gate = 2.0 * normed[0];
    let silu = gate / (1.0 + (-gate).exp());
    let hidden = plus(&EMBED[token], &[0.0, 0.0, silu * normed[1], 0.0]);
    logits_for(&hidden, &OUTPUT)
}

#[test]
fn feed_forward_adds_the_gated_projection() {
    let logits = model(&gated(quiet(), 0).build()).forward(6, 0).unwrap();
    assert_close(&logits, &gated_expectation(6));
}

#[test]
fn every_block_runs_with_its_own_cache_layer() {
    let source = gated(zero_block(quiet(), 1), 1)
        .with_metadata("llama.block_count", GGUFValue::Uint32(2))
        .build();
    let mut llama = model(&source);
    assert_close(&llama.forward(2, 0).unwrap(), &gated_expectation(2));
    assert_close(&llama.forward(2, 1).unwrap(), &gated_expectation(2));
}
