// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::FeedForward;
use crate::config::{Activation, DecoderConfig};
use crate::rope::RopeStyle;
use crate::test_support::{assert_close, matrix, normed, picker, tiny_config, zero_block};
use crate::weights::BlockWeights;

const SUB_NORM: [f32; 8] = [2.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
const INPUTS: [[f32; 4]; 2] = [[-1.5, 2.0, 3.0, -0.5], [2.0, -1.0, 0.5, 4.0]];

fn config(activation: Activation) -> DecoderConfig {
    DecoderConfig {
        activation,
        ..tiny_config(RopeStyle::Adjacent)
    }
}

/// Hidden row `[a(x0) * x2, a(x1) * x3, 0, ...]` projected to `[h0, h1, 0, 0]`.
fn block(sub_norm: Option<&[f32]>) -> BlockWeights {
    BlockWeights {
        ffn_gate: matrix(picker(8, 4, &[(0, 0), (1, 1)])),
        ffn_up: matrix(picker(8, 4, &[(0, 2), (1, 3)])),
        ffn_down: matrix(picker(4, 8, &[(0, 0), (1, 1)])),
        ffn_sub_norm: sub_norm.map(<[f32]>::to_vec),
        ..zero_block(&config(Activation::Silu))
    }
}

fn run(activation: Activation, block: &BlockWeights) -> Vec<f32> {
    let input = INPUTS.as_flattened();
    let mut output = vec![0.0; input.len()];
    FeedForward::new(&config(activation)).forward(block, input, &mut output);
    output
}

fn hidden(row: &[f32; 4], activate: fn(f32) -> f32) -> Vec<f32> {
    let mut hidden = vec![0.0; 8];
    hidden[0] = activate(row[0]) * row[2];
    hidden[1] = activate(row[1]) * row[3];
    hidden
}

fn squared_relu(value: f32) -> f32 {
    value.max(0.0) * value.max(0.0)
}

fn expected(sub_norm: Option<&[f32]>) -> Vec<f32> {
    INPUTS
        .iter()
        .flat_map(|row| {
            let mut hidden = hidden(row, squared_relu);
            if let Some(weight) = sub_norm {
                hidden = normed(&hidden)
                    .iter()
                    .zip(weight)
                    .map(|(n, w)| n * w)
                    .collect();
            }
            [hidden[0], hidden[1], 0.0, 0.0]
        })
        .collect()
}

#[test]
fn squared_relu_replaces_silu_on_the_gate() {
    let output = run(Activation::ReluSquared, &block(None));
    assert_close(&output, &expected(None));
    assert_eq!(output[0].to_bits(), 0.0_f32.to_bits());
}

#[test]
fn ffn_sub_norm_normalizes_each_gated_row_before_the_down_projection() {
    let output = run(Activation::ReluSquared, &block(Some(&SUB_NORM)));
    assert_close(&output, &expected(Some(&SUB_NORM)));
}
