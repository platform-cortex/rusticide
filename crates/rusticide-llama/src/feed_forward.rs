// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_kernels::{mul_assign, silu_assign};
use rusticide_model::matmul_batch;

use crate::config::LlamaConfig;
use crate::layout::to_token_major;
use crate::weights::BlockWeights;

/// The `SwiGLU` feed-forward network for a batch of tokens, with reusable
/// scratch buffers. `gate`, `up` and `down` are laid out by weight row.
#[derive(Debug)]
pub(crate) struct FeedForward {
    width: usize,
    length: usize,
    gate: Vec<f32>,
    up: Vec<f32>,
    hidden: Vec<f32>,
    down: Vec<f32>,
}

impl FeedForward {
    pub(crate) const fn new(config: &LlamaConfig) -> Self {
        Self {
            width: config.embedding_length,
            length: config.feed_forward_length,
            gate: Vec::new(),
            up: Vec::new(),
            hidden: Vec::new(),
            down: Vec::new(),
        }
    }

    /// Writes `ffn_down(silu(ffn_gate(input)) * ffn_up(input))` for the
    /// tokens of `input`, one row each.
    pub(crate) fn forward(&mut self, block: &BlockWeights, input: &[f32], output: &mut [f32]) {
        let tokens = input.len() / self.width;
        let len = self.length * tokens;
        for buffer in [&mut self.gate, &mut self.up, &mut self.hidden] {
            buffer.resize(len, 0.0);
        }
        matmul_batch(
            input,
            tokens,
            &mut [
                (&block.ffn_gate, &mut self.gate),
                (&block.ffn_up, &mut self.up),
            ],
        );
        silu_assign(&mut self.gate);
        mul_assign(&mut self.gate, &self.up);
        to_token_major(&self.gate, tokens, &mut self.hidden);
        self.down.resize(input.len(), 0.0);
        matmul_batch(
            &self.hidden,
            tokens,
            &mut [(&block.ffn_down, &mut self.down)],
        );
        to_token_major(&self.down, tokens, output);
    }
}
