// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_model::{ModelError, Result};

use crate::rope::RopeStyle;

/// The `InvalidConfig` reason for dimensions whose products overflow `usize`.
pub const DIMENSIONS_TOO_LARGE: &str = "model dimensions are too large";

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
                reason: DIMENSIONS_TOO_LARGE,
            })
    }

    /// Checks the sizes the decoder divides by, rotates in pairs or splits
    /// rows by.
    ///
    /// # Errors
    ///
    /// `InvalidConfig` when the head count or head dimension is zero, the
    /// head dimension is odd, the kv head count is zero or does not divide
    /// the head count, the embedding length is not the head count times
    /// the head dimension, or the block count, feed-forward length or
    /// context length is zero.
    pub fn validate(&self) -> Result<()> {
        ensure(self.head_count != 0, "head count must be non-zero")?;
        ensure(
            self.head_count_kv != 0 && self.head_count.is_multiple_of(self.head_count_kv),
            "kv head count must be non-zero and divide the head count",
        )?;
        ensure(
            self.head_dim != 0 && self.head_dim.is_multiple_of(2),
            "head dimension must be even and non-zero",
        )?;
        let width = self.head_count.checked_mul(self.head_dim);
        ensure(
            matches!(width, Some(width) if width == self.embedding_length),
            "embedding length must be divisible by the head count: it must equal the head count times the head dimension",
        )?;
        ensure(
            self.block_count != 0 && self.feed_forward_length != 0 && self.context_length != 0,
            "block count, feed-forward length and context length must be non-zero",
        )
    }
}

const fn ensure(condition: bool, reason: &'static str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(ModelError::InvalidConfig { reason })
    }
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
