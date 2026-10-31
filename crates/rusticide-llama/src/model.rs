// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_kernels::{add_assign, mul_assign, rms_norm};
use rusticide_model::{KvCache, Model, ModelError, ModelSource, Result};

use crate::attention::Attention;
use crate::config::LlamaConfig;
use crate::feed_forward::FeedForward;
use crate::variant::Variant;
use crate::weights::LlamaWeights;

/// A llama model running one token at a time over a KV cache.
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
        let width = config.embedding_length;
        Ok(Self {
            cache: KvCache::new(config.block_count, config.kv_dim()?, config.context_length),
            attention: Attention::new(&config)?,
            feed_forward: FeedForward::new(&config),
            hidden: vec![0.0; width],
            normed: vec![0.0; width],
            delta: vec![0.0; width],
            config,
            weights,
        })
    }
}

/// Writes `rms_norm(input) * weight` into `output`.
fn normalize(input: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    output.copy_from_slice(input);
    rms_norm(output, eps);
    mul_assign(output, weight);
}

impl Model for Llama {
    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn context_length(&self) -> usize {
        self.config.context_length
    }

    fn forward(&mut self, token: u32, position: usize) -> Result<Vec<f32>> {
        let embedding = usize::try_from(token)
            .ok()
            .and_then(|row| self.weights.token_embd.row(row))
            .ok_or(ModelError::InvalidConfig {
                reason: "token out of range",
            })?;
        self.hidden.copy_from_slice(embedding);
        let eps = self.config.rms_norm_eps;
        for (layer, block) in self.weights.blocks.iter().enumerate() {
            normalize(&self.hidden, &block.attn_norm, eps, &mut self.normed);
            self.attention.forward(
                block,
                &mut self.cache,
                layer,
                position,
                &self.normed,
                &mut self.delta,
            )?;
            add_assign(&mut self.hidden, &self.delta);
            normalize(&self.hidden, &block.ffn_norm, eps, &mut self.normed);
            self.feed_forward
                .forward(block, &self.normed, &mut self.delta);
            add_assign(&mut self.hidden, &self.delta);
        }
        normalize(
            &self.hidden,
            &self.weights.output_norm,
            eps,
            &mut self.normed,
        );
        let output = self
            .weights
            .output
            .as_ref()
            .unwrap_or(&self.weights.token_embd);
        let mut logits = vec![0.0; self.config.vocab_size];
        output.matvec(&self.normed, &mut logits);
        Ok(logits)
    }

    fn reset(&mut self) {
        self.cache.clear();
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
#[path = "model_error_tests.rs"]
mod error_tests;
