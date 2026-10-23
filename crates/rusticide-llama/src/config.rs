// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_io::{GGUFValue, Metadata};
use rusticide_model::{ModelError, ModelSource, Result};

/// Hyperparameters of a llama model, validated against its source.
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaConfig {
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

const TOO_LARGE: &str = "model dimensions are too large";
const ROPE_DIMENSION_KEY: &str = "llama.rope.dimension_count";
const DEFAULT_ROPE_FREQ_BASE: f32 = 10_000.0;
const DEFAULT_RMS_NORM_EPS: f32 = 1e-5;

fn to_usize(value: u64) -> Result<usize> {
    usize::try_from(value).map_err(|_| ModelError::InvalidConfig { reason: TOO_LARGE })
}

fn required(value: Option<u64>, key: &str) -> Result<usize> {
    to_usize(value.ok_or_else(|| ModelError::MissingMetadata {
        key: key.to_owned(),
    })?)
}

const fn ensure(condition: bool, reason: &'static str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(ModelError::InvalidConfig { reason })
    }
}

fn ensure_rope_dimension(metadata: &Metadata, head_dim: usize) -> Result<()> {
    let Some(value) = metadata.get(ROPE_DIMENSION_KEY) else {
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
    pub fn from_source(source: &dyn ModelSource) -> Result<Self> {
        let metadata = source.metadata();
        let context_length = required(metadata.context_length(), "llama.context_length")?;
        let embedding_length = required(metadata.embedding_length(), "llama.embedding_length")?;
        let block_count = required(metadata.block_count(), "llama.block_count")?;
        let feed_forward_length =
            required(metadata.feed_forward_length(), "llama.feed_forward_length")?;
        let head_count = required(metadata.head_count(), "llama.attention.head_count")?;
        let head_count_kv = metadata.head_count_kv().map_or(Ok(head_count), to_usize)?;
        ensure(head_count != 0, "head count must be non-zero")?;
        ensure(
            embedding_length % head_count == 0,
            "embedding length must be divisible by the head count",
        )?;
        ensure(
            head_count_kv != 0 && head_count % head_count_kv == 0,
            "kv head count must be non-zero and divide the head count",
        )?;
        let head_dim = embedding_length / head_count;
        ensure(
            head_dim != 0 && head_dim % 2 == 0,
            "head dimension must be even and non-zero",
        )?;
        ensure(
            block_count != 0 && feed_forward_length != 0 && context_length != 0,
            "block count, feed-forward length and context length must be non-zero",
        )?;
        ensure(
            sizes_fit(
                embedding_length,
                feed_forward_length,
                head_count_kv,
                head_dim,
            ),
            TOO_LARGE,
        )?;
        ensure_rope_dimension(metadata, head_dim)?;
        Ok(Self {
            vocab_size: vocab_size(source, embedding_length)?,
            embedding_length,
            block_count,
            feed_forward_length,
            head_count,
            head_count_kv,
            head_dim,
            context_length,
            rope_freq_base: metadata.rope_freq_base().unwrap_or(DEFAULT_ROPE_FREQ_BASE),
            rms_norm_eps: metadata.rms_norm_eps().unwrap_or(DEFAULT_RMS_NORM_EPS),
        })
    }

    /// Width of one key or value row: `head_count_kv * head_dim`.
    ///
    /// # Errors
    ///
    /// `InvalidConfig` when the product overflows.
    pub fn kv_dim(&self) -> Result<usize> {
        self.head_count_kv
            .checked_mul(self.head_dim)
            .ok_or(ModelError::InvalidConfig { reason: TOO_LARGE })
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
