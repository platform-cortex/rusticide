// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_kernels::{dot, rope_adjacent, softmax};
use rusticide_model::{KvCache, Result};

use crate::config::LlamaConfig;
use crate::weights::BlockWeights;

/// Grouped-query self-attention for one token, with reusable scratch buffers.
#[derive(Debug)]
pub(crate) struct Attention {
    head_dim: usize,
    group: usize,
    kv_dim: usize,
    scale: f32,
    rope_freq_base: f32,
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    scores: Vec<f32>,
    mixed: Vec<f32>,
}

impl Attention {
    pub(crate) fn new(config: &LlamaConfig) -> Result<Self> {
        let kv_dim = config.kv_dim()?;
        #[expect(
            clippy::cast_precision_loss,
            reason = "head dimensions are far below 2^24, where f32 is exact"
        )]
        let head_dim = config.head_dim as f32;
        Ok(Self {
            head_dim: config.head_dim,
            group: config.head_count / config.head_count_kv,
            kv_dim,
            scale: head_dim.sqrt(),
            rope_freq_base: config.rope_freq_base,
            query: vec![0.0; config.embedding_length],
            key: vec![0.0; kv_dim],
            value: vec![0.0; kv_dim],
            scores: Vec::new(),
            mixed: vec![0.0; config.embedding_length],
        })
    }

    /// Writes `attn_output(attention(input))` for the token at `position`,
    /// storing its key and value in `cache` under `layer`.
    pub(crate) fn forward(
        &mut self,
        block: &BlockWeights,
        cache: &mut KvCache,
        layer: usize,
        position: usize,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        block.attn_q.matvec(input, &mut self.query);
        block.attn_k.matvec(input, &mut self.key);
        block.attn_v.matvec(input, &mut self.value);
        let pairs = self.head_dim / 2;
        for head in self
            .query
            .chunks_exact_mut(self.head_dim)
            .chain(self.key.chunks_exact_mut(self.head_dim))
        {
            rope_adjacent(head, position, self.rope_freq_base, pairs);
        }
        cache.store(layer, position, &self.key, &self.value)?;
        let (keys, values) = (cache.keys(layer), cache.values(layer));
        for (head, (query, mixed)) in self
            .query
            .chunks_exact(self.head_dim)
            .zip(self.mixed.chunks_exact_mut(self.head_dim))
            .enumerate()
        {
            let offset = (head / self.group) * self.head_dim;
            let span = offset..offset + self.head_dim;
            self.scores.clear();
            self.scores.extend(
                keys.chunks_exact(self.kv_dim)
                    .map(|key| dot(query, &key[span.clone()]) / self.scale),
            );
            softmax(&mut self.scores);
            mixed.fill(0.0);
            for (&weight, value) in self.scores.iter().zip(values.chunks_exact(self.kv_dim)) {
                for (slot, &entry) in mixed.iter_mut().zip(&value[span.clone()]) {
                    *slot = entry.mul_add(weight, *slot);
                }
            }
        }
        block.attn_output.matvec(&self.mixed, output);
        Ok(())
    }
}

#[cfg(test)]
#[path = "attention_tests.rs"]
mod tests;
