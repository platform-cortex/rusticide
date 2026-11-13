// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_model::{ModelSource, Result};
use rusticide_transformer::DecoderConfig;

use crate::variant::Variant;

/// Hyperparameters of a llama model, validated against its source.
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaConfig {
    pub variant: Variant,
    pub vocab_size: usize,
    pub embedding_length: usize,
    pub block_count: usize,
    pub feed_forward_length: usize,
    pub head_count: usize,
    pub head_count_kv: usize,
    pub head_dim: usize,
    pub context_length: usize,
    pub rope_freq_base: f32,
    pub rms_norm_eps: f32,
}

impl LlamaConfig {
    /// Reads and validates the configuration; `vocab_size` comes from the
    /// first dimension of `token_embd.weight`.
    ///
    /// # Errors
    ///
    /// As [`DecoderConfig::from_source`].
    pub fn from_source(source: &dyn ModelSource, variant: Variant) -> Result<Self> {
        let config = DecoderConfig::from_source(source, &variant.family())?;
        Ok(Self {
            variant,
            vocab_size: config.vocab_size,
            embedding_length: config.embedding_length,
            block_count: config.block_count,
            feed_forward_length: config.feed_forward_length,
            head_count: config.head_count,
            head_count_kv: config.head_count_kv,
            head_dim: config.head_dim,
            context_length: config.context_length,
            rope_freq_base: config.rope_freq_base,
            rms_norm_eps: config.rms_norm_eps,
        })
    }

    /// The family-neutral decoder hyperparameters of this model.
    #[must_use]
    pub const fn decoder(&self) -> DecoderConfig {
        DecoderConfig {
            embedding_length: self.embedding_length,
            block_count: self.block_count,
            feed_forward_length: self.feed_forward_length,
            head_count: self.head_count,
            head_count_kv: self.head_count_kv,
            head_dim: self.head_dim,
            context_length: self.context_length,
            vocab_size: self.vocab_size,
            rope_freq_base: self.rope_freq_base,
            rms_norm_eps: self.rms_norm_eps,
            rope: self.variant.family().rope,
            activation: self.variant.family().activation,
        }
    }
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "config_qwen2_tests.rs"]
mod qwen2_tests;
