// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_io::{GGUFValue, Metadata};
use rusticide_model::{ModelError, ModelSource, Result};
use rusticide_transformer::{Activation, DIMENSIONS_TOO_LARGE, DecoderConfig};

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

const DEFAULT_ROPE_FREQ_BASE: f32 = 10_000.0;
const DEFAULT_RMS_NORM_EPS: f32 = 1e-5;

fn to_usize(value: u64) -> Result<usize> {
    usize::try_from(value).map_err(|_| ModelError::InvalidConfig {
        reason: DIMENSIONS_TOO_LARGE,
    })
}

fn required(value: Option<u64>, variant: Variant, suffix: &str) -> Result<usize> {
    to_usize(value.ok_or_else(|| ModelError::MissingMetadata {
        key: format!("{}.{suffix}", variant.architecture()),
    })?)
}

const fn ensure(condition: bool, reason: &'static str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(ModelError::InvalidConfig { reason })
    }
}

fn ensure_rope_dimension(metadata: &Metadata, variant: Variant, head_dim: usize) -> Result<()> {
    let key = format!("{}.rope.dimension_count", variant.architecture());
    let Some(value) = metadata.get(&key) else {
        return Ok(());
    };
    let dimension = match *value {
        GGUFValue::Uint32(n) => u64::from(n),
        GGUFValue::Uint64(n) => n,
        _ => {
            return Err(ModelError::InvalidConfig {
                reason: "rope dimension count must be an unsigned integer",
            });
        }
    };
    ensure(
        usize::try_from(dimension) == Ok(head_dim),
        "rope dimension count must equal the head dimension",
    )
}

fn vocab_size(source: &dyn ModelSource, embedding_length: usize) -> Result<usize> {
    let shape =
        source
            .tensor_shape("token_embd.weight")
            .ok_or_else(|| ModelError::MissingTensor {
                name: "token_embd.weight".to_owned(),
            })?;
    match shape.as_slice() {
        [vocab, embedding] if *embedding == embedding_length && *vocab > 0 => Ok(*vocab),
        _ => Err(ModelError::InvalidConfig {
            reason: "token_embd.weight must have shape [vocab, embedding length]",
        }),
    }
}

impl LlamaConfig {
    /// Reads and validates the configuration; `vocab_size` comes from the
    /// first dimension of `token_embd.weight`.
    ///
    /// # Errors
    ///
    /// `MissingMetadata` for an absent required key, `MissingTensor` without
    /// `token_embd.weight`, and `InvalidConfig` for any inconsistent or
    /// overflowing dimension.
    pub fn from_source(source: &dyn ModelSource, variant: Variant) -> Result<Self> {
        let mut config = Self::read(source.metadata(), variant)?;
        config.decoder().validate()?;
        ensure(
            sizes_fit(
                config.embedding_length,
                config.feed_forward_length,
                config.head_count_kv,
                config.head_dim,
            ),
            DIMENSIONS_TOO_LARGE,
        )?;
        ensure_rope_dimension(source.metadata(), variant, config.head_dim)?;
        config.vocab_size = vocab_size(source, config.embedding_length)?;
        Ok(config)
    }

    /// The hyperparameters as `metadata` states them, unchecked; `vocab_size`
    /// is left zero.
    fn read(metadata: &Metadata, variant: Variant) -> Result<Self> {
        let embedding_length = required(metadata.embedding_length(), variant, "embedding_length")?;
        let head_count = required(metadata.head_count(), variant, "attention.head_count")?;
        Ok(Self {
            variant,
            vocab_size: 0,
            embedding_length,
            block_count: required(metadata.block_count(), variant, "block_count")?,
            feed_forward_length: required(
                metadata.feed_forward_length(),
                variant,
                "feed_forward_length",
            )?,
            head_count,
            head_count_kv: metadata.head_count_kv().map_or(Ok(head_count), to_usize)?,
            head_dim: embedding_length.checked_div(head_count).unwrap_or(0),
            context_length: required(metadata.context_length(), variant, "context_length")?,
            rope_freq_base: metadata.rope_freq_base().unwrap_or(DEFAULT_ROPE_FREQ_BASE),
            rms_norm_eps: metadata.rms_norm_eps().unwrap_or(DEFAULT_RMS_NORM_EPS),
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
            rope: self.variant.rope(),
            activation: Activation::Silu,
        }
    }
}

fn sizes_fit(embedding: usize, feed_forward: usize, kv_heads: usize, head_dim: usize) -> bool {
    let projection = kv_heads
        .checked_mul(head_dim)
        .and_then(|kv_dim| kv_dim.checked_mul(embedding));
    projection.is_some()
        && embedding.checked_mul(embedding).is_some()
        && embedding.checked_mul(feed_forward).is_some()
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "config_qwen2_tests.rs"]
mod qwen2_tests;
