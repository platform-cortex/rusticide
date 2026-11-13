// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_core::Tensor;
use rusticide_model::Model;

use super::Decoder;
use crate::config::DecoderConfig;
use crate::rope::RopeStyle;
use crate::test_support::{
    assert_close, identity, matrix, normed, picker, tensor, tiny_config, zero_block,
};
use crate::weights::{BlockWeights, DecoderWeights};

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

/// The tiny decoder with literal embedding and output and every block
/// weight zero, after `edit` changes its weights.
fn quiet(edit: impl FnOnce(&DecoderConfig, &mut DecoderWeights)) -> Decoder {
    let config = tiny_config(RopeStyle::Adjacent);
    let mut weights = DecoderWeights {
        token_embd: matrix(rows(&EMBED)),
        blocks: vec![zero_block(&config)],
        output_norm: OUTPUT_NORM.to_vec(),
        output: Some(matrix(rows(&OUTPUT))),
    };
    edit(&config, &mut weights);
    let config = DecoderConfig {
        block_count: weights.blocks.len(),
        ..config
    };
    Decoder::new(config, weights).unwrap()
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
    let logits = quiet(|_, _| {}).forward(0, 0).unwrap();
    assert_close(&logits, &logits_for(&EMBED[0], &OUTPUT));
}

#[test]
fn missing_output_weight_ties_to_the_embedding() {
    let mut decoder = quiet(|_, weights| weights.output = None);
    let logits = decoder.forward(3, 0).unwrap();
    assert_close(&logits, &logits_for(&EMBED[3], &EMBED));
}

#[test]
fn second_position_averages_values_when_queries_are_zero() {
    let mut decoder = quiet(|_, weights| {
        let block = &mut weights.blocks[0];
        block.attn_k = matrix(picker(2, 4, &[(0, 2), (1, 3)]));
        block.attn_v = matrix(picker(2, 4, &[(0, 0), (1, 1)]));
        block.attn_output = matrix(identity(4));
    });
    decoder.forward(1, 0).unwrap();
    let logits = decoder.forward(4, 1).unwrap();
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
    let mut decoder = quiet(|_, weights| {
        let block = &mut weights.blocks[0];
        block.attn_norm = attn_norm.to_vec();
        block.attn_q = matrix(identity(4));
        block.attn_v = matrix(picker(2, 4, &[(0, 0), (1, 1)]));
        block.attn_output = matrix(identity(4));
    });
    let logits = decoder.forward(5, 0).unwrap();
    let normed = normed(&EMBED[5]);
    let value = [normed[0] * attn_norm[0], normed[1] * attn_norm[1]];
    let hidden = plus(&EMBED[5], &[value[0], value[1], value[0], value[1]]);
    assert_close(&logits, &logits_for(&hidden, &OUTPUT));
}

fn gate(block: &mut BlockWeights) {
    block.ffn_norm = vec![2.0, 1.0, 1.0, 1.0];
    block.ffn_gate = matrix(picker(8, 4, &[(0, 0)]));
    block.ffn_up = matrix(picker(8, 4, &[(0, 1)]));
    block.ffn_down = matrix(picker(4, 8, &[(2, 0)]));
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
    let mut decoder = quiet(|_, weights| gate(&mut weights.blocks[0]));
    let logits = decoder.forward(6, 0).unwrap();
    assert_close(&logits, &gated_expectation(6));
}

#[test]
fn every_block_runs_with_its_own_cache_layer() {
    let mut decoder = quiet(|config, weights| {
        let mut second = zero_block(config);
        gate(&mut second);
        weights.blocks.push(second);
    });
    assert_close(&decoder.forward(2, 0).unwrap(), &gated_expectation(2));
    assert_close(&decoder.forward(2, 1).unwrap(), &gated_expectation(2));
}
