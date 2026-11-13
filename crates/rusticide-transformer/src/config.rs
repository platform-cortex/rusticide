// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_model::{ModelError, Result};

use crate::rope::RopeStyle;

/// The activation of the gated feed-forward.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Silu,
}

/// Hyperparameters of a decoder-only transformer, shared by every family.
#[derive(Debug, Clone, PartialEq)]
pub struct DecoderConfig {
    pub embedding_length: usize,
    pub block_count: usize,
    pub feed_forward_length: usize,
    pub head_count: usize,
    pub head_count_kv: usize,
    pub head_dim: usize,
    pub context_length: usize,
    pub vocab_size: usize,
    pub rope_freq_base: f32,
    pub rms_norm_eps: f32,
    pub rope: RopeStyle,
    pub activation: Activation,
}

impl DecoderConfig {
    /// Width of one key or value row: `head_count_kv * head_dim`.
    ///
    /// # Errors
    ///
    /// `InvalidConfig` when the product overflows.
    pub fn kv_dim(&self) -> Result<usize> {
        self.head_count_kv
            .checked_mul(self.head_dim)
            .ok_or(ModelError::InvalidConfig {
                reason: "model dimensions are too large",
            })
    }
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
