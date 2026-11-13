// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_kernels::{matmul_batch, mul_assign, relu_squared_assign, silu_assign};

use crate::batch::zip_rows;
use crate::config::{Activation, DecoderConfig};
use crate::layout::to_token_major;
use crate::norm::normalize_rows;
use crate::weights::BlockWeights;

/// The gated feed-forward network for a batch of tokens, with reusable
/// scratch buffers. `gate`, `up` and `down` are laid out by weight row.
#[derive(Debug)]
pub struct FeedForward {
    width: usize,
    length: usize,
    activation: Activation,
    eps: f32,
    gate: Vec<f32>,
    up: Vec<f32>,
    hidden: Vec<f32>,
    down: Vec<f32>,
}

impl FeedForward {
    #[must_use]
    pub const fn new(config: &DecoderConfig) -> Self {
        Self {
            width: config.embedding_length,
            length: config.feed_forward_length,
            activation: config.activation,
            eps: config.rms_norm_eps,
            gate: Vec::new(),
            up: Vec::new(),
            hidden: Vec::new(),
            down: Vec::new(),
        }
    }

    /// Writes `ffn_down(sub_norm(act(ffn_gate(input)) * ffn_up(input)))` for
    /// the tokens of `input`, one row each; `act` is the configured
    /// activation and the sub-norm applies only when the block has one.
    pub fn forward(&mut self, block: &BlockWeights, input: &[f32], output: &mut [f32]) {
        let tokens = input.len() / self.width;
        let len = self.length * tokens;
        for buffer in [&mut self.gate, &mut self.up, &mut self.hidden] {
            buffer.resize(len, 0.0);
        }
        matmul_batch(
            input,
            tokens,
            &mut [
                (block.ffn_gate.weights(), &mut self.gate),
                (block.ffn_up.weights(), &mut self.up),
            ],
        );
        let activation = self.activation;
        zip_rows(&mut self.gate, &self.up, self.length, tokens, |gate, up| {
            match activation {
                Activation::Silu => silu_assign(gate),
                Activation::ReluSquared => relu_squared_assign(gate),
            }
            mul_assign(gate, up);
        });
        to_token_major(&self.gate, tokens, &mut self.hidden);
        if let Some(sub_norm) = &block.ffn_sub_norm {
            normalize_rows(&self.hidden, sub_norm, self.eps, tokens, &mut self.gate);
            std::mem::swap(&mut self.hidden, &mut self.gate);
        }
        self.down.resize(input.len(), 0.0);
        matmul_batch(
            &self.hidden,
            tokens,
            &mut [(block.ffn_down.weights(), &mut self.down)],
        );
        to_token_major(&self.down, tokens, output);
    }
}

#[cfg(test)]
#[path = "feed_forward_tests.rs"]
mod tests;
