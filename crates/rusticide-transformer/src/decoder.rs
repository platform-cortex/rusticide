// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_kernels::add_assign;
use rusticide_model::{KvCache, Model, ModelError, Result};

use crate::attention::Attention;
use crate::batch::zip_rows;
use crate::config::DecoderConfig;
use crate::feed_forward::FeedForward;
use crate::norm::{normalize, normalize_rows};
use crate::weights::DecoderWeights;

/// A decoder-only transformer running batches of tokens over a KV cache.
#[derive(Debug)]
pub struct Decoder {
    config: DecoderConfig,
    weights: DecoderWeights,
    cache: KvCache,
    attention: Attention,
    feed_forward: FeedForward,
    hidden: Vec<f32>,
    normed: Vec<f32>,
    delta: Vec<f32>,
}

impl Decoder {
    /// # Errors
    ///
    /// `InvalidConfig` when [`DecoderConfig::validate`] rejects `config` or
    /// the key-value width overflows.
    pub fn new(config: DecoderConfig, weights: DecoderWeights) -> Result<Self> {
        let attention = Attention::new(&config)?;
        Ok(Self {
            cache: KvCache::new(config.block_count, config.kv_dim()?, config.context_length),
            attention,
            feed_forward: FeedForward::new(&config),
            hidden: Vec::new(),
            normed: Vec::new(),
            delta: Vec::new(),
            config,
            weights,
        })
    }
}

impl Model for Decoder {
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

impl Decoder {
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

#[cfg(test)]
#[path = "decoder_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "decoder_batch_tests.rs"]
mod batch_tests;

#[cfg(test)]
#[path = "decoder_error_tests.rs"]
mod error_tests;

#[cfg(test)]
#[path = "decoder_attention_tests.rs"]
mod attention_tests;
