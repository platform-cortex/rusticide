// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_kernels::add_assign;
use rusticide_model::{KvCache, Model, ModelError, ModelSource, Result};
use rusticide_transformer::{Attention, FeedForward, normalize, normalize_rows, zip_rows};

use crate::config::LlamaConfig;
use crate::variant::Variant;
use crate::weights::LlamaWeights;

/// A llama model running batches of tokens over a KV cache.
#[derive(Debug)]
pub struct Llama {
    config: LlamaConfig,
    weights: LlamaWeights,
    cache: KvCache,
    attention: Attention,
    feed_forward: FeedForward,
    hidden: Vec<f32>,
    normed: Vec<f32>,
    delta: Vec<f32>,
}

impl Llama {
    /// Reads the configuration and weights of `source`.
    ///
    /// # Errors
    ///
    /// Any error from [`LlamaConfig::from_source`] or [`LlamaWeights::load`].
    pub fn from_source(source: &dyn ModelSource, variant: Variant) -> Result<Self> {
        let config = LlamaConfig::from_source(source, variant)?;
        let weights = LlamaWeights::load(source, &config)?;
        let decoder = config.decoder();
        Ok(Self {
            cache: KvCache::new(config.block_count, decoder.kv_dim()?, config.context_length),
            attention: Attention::new(&decoder)?,
            feed_forward: FeedForward::new(&decoder),
            hidden: Vec::new(),
            normed: Vec::new(),
            delta: Vec::new(),
            config,
            weights,
        })
    }
}

impl Model for Llama {
    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn context_length(&self) -> usize {
        self.config.context_length
    }

    fn forward(&mut self, token: u32, position: usize) -> Result<Vec<f32>> {
        self.forward_batch(&[token], position)
    }

    fn forward_batch(&mut self, tokens: &[u32], start: usize) -> Result<Vec<f32>> {
        // On a pool thread every parallel region below is a local fork-join;
        // from outside the pool each one is a cross-thread injection and
        // wake-up, about 2.5x the cost on an M1 Pro.
        rayon::scope(|_| self.forward_on_pool(tokens, start))
    }

    fn reset(&mut self) {
        self.cache.clear();
    }
}

impl Llama {
    fn forward_on_pool(&mut self, tokens: &[u32], start: usize) -> Result<Vec<f32>> {
        self.embed(tokens, start)?;
        let (eps, width) = (self.config.rms_norm_eps, self.config.embedding_length);
        let count = tokens.len();
        self.normed.resize(self.hidden.len(), 0.0);
        self.delta.resize(self.hidden.len(), 0.0);
        for (layer, block) in self.weights.blocks.iter().enumerate() {
            normalize_rows(&self.hidden, &block.attn_norm, eps, count, &mut self.normed);
            self.attention.forward(
                block,
                &mut self.cache,
                layer,
                start,
                &self.normed,
                &mut self.delta,
            )?;
            zip_rows(&mut self.hidden, &self.delta, width, count, add_assign);
            normalize_rows(&self.hidden, &block.ffn_norm, eps, count, &mut self.normed);
            self.feed_forward
                .forward(block, &self.normed, &mut self.delta);
            zip_rows(&mut self.hidden, &self.delta, width, count, add_assign);
        }
        let last = self.hidden.len().saturating_sub(width);
        let (hidden, normed) = (&self.hidden[last..], &mut self.normed[..width]);
        normalize(hidden, &self.weights.output_norm, eps, normed);
        let output = self
            .weights
            .output
            .as_ref()
            .unwrap_or(&self.weights.token_embd);
        let mut logits = vec![0.0; self.config.vocab_size];
        output.matvec(normed, &mut logits);
        Ok(logits)
    }

    /// Loads one embedding row per token into `hidden`, after checking the
    /// batch fits the context, so a rejected batch leaves the cache as is.
    fn embed(&mut self, tokens: &[u32], start: usize) -> Result<()> {
        if tokens.is_empty() {
            return Err(ModelError::InvalidConfig {
                reason: "no tokens to forward",
            });
        }
        let limit = self.config.context_length;
        if start
            .checked_add(tokens.len())
            .is_none_or(|end| end > limit)
        {
            return Err(ModelError::ContextFull { limit });
        }
        let width = self.config.embedding_length;
        self.hidden.resize(tokens.len() * width, 0.0);
        for (&token, row) in tokens.iter().zip(self.hidden.chunks_exact_mut(width)) {
            let found = usize::try_from(token)
                .is_ok_and(|index| self.weights.token_embd.copy_row(index, row));
            if !found {
                return Err(ModelError::InvalidConfig {
                    reason: "token out of range",
                });
            }
        }
        Ok(())
    }
}

/// The registry [`rusticide_model::Loader`] for the `llama` architecture.
///
/// # Errors
///
/// Any error from [`Llama::from_source`].
pub fn load(source: &dyn ModelSource) -> Result<Box<dyn Model>> {
    Ok(Box::new(Llama::from_source(source, Variant::Llama)?))
}

/// The registry [`rusticide_model::Loader`] for the `qwen2` architecture.
///
/// # Errors
///
/// Any error from [`Llama::from_source`].
pub fn load_qwen2(source: &dyn ModelSource) -> Result<Box<dyn Model>> {
    Ok(Box::new(Llama::from_source(source, Variant::Qwen2)?))
}

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "model_batch_tests.rs"]
mod batch_tests;

#[cfg(test)]
#[path = "model_error_tests.rs"]
mod error_tests;

#[cfg(test)]
#[path = "model_attention_tests.rs"]
mod attention_tests;
