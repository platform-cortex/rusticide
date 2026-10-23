// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_kernels::{mul_assign, silu};

use crate::config::LlamaConfig;
use crate::weights::BlockWeights;

/// The `SwiGLU` feed-forward network, with reusable scratch buffers.
#[derive(Debug)]
pub(crate) struct FeedForward {
    gate: Vec<f32>,
    up: Vec<f32>,
}

impl FeedForward {
    pub(crate) fn new(config: &LlamaConfig) -> Self {
        Self {
            gate: vec![0.0; config.feed_forward_length],
            up: vec![0.0; config.feed_forward_length],
        }
    }

    /// Writes `ffn_down(silu(ffn_gate(input)) * ffn_up(input))`.
    pub(crate) fn forward(&mut self, block: &BlockWeights, input: &[f32], output: &mut [f32]) {
        block.ffn_gate.matvec(input, &mut self.gate);
        block.ffn_up.matvec(input, &mut self.up);
        for value in &mut self.gate {
            *value = silu(*value);
        }
        mul_assign(&mut self.gate, &self.up);
        block.ffn_down.matvec(&self.gate, output);
    }
}
