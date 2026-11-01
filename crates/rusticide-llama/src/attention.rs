// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rayon::prelude::*;
use rusticide_kernels::{add_assign, dot_rows, matmul_batch, mix_rows, softmax};
use rusticide_model::{KvCache, Result};

use crate::config::LlamaConfig;
use crate::layout::to_token_major;
use crate::variant::RopeStyle;
use crate::weights::BlockWeights;

/// The head layout of grouped-query attention.
#[derive(Debug, Clone, Copy)]
struct Heads {
    head_dim: usize,
    group: usize,
    kv_dim: usize,
    scale: f32,
    rope_freq_base: f32,
    rope: RopeStyle,
}

impl Heads {
    /// Rotates every head of `query` and `key` to `position`.
    fn rotate(&self, query: &mut [f32], key: &mut [f32], position: usize) {
        let pairs = self.head_dim / 2;
        for head in query
            .chunks_exact_mut(self.head_dim)
            .chain(key.chunks_exact_mut(self.head_dim))
        {
            self.rope.apply(head, position, self.rope_freq_base, pairs);
        }
    }

    /// Writes into `mixed` query head `head` attending over every row of
    /// `keys` and `values`.
    fn mix(
        &self,
        head: usize,
        query: &[f32],
        cache: (&[f32], &[f32]),
        scores: &mut Vec<f32>,
        mixed: &mut [f32],
    ) {
        let (keys, values) = cache;
        let offset = (head / self.group) * self.head_dim;
        scores.clear();
        scores.resize(keys.len() / self.kv_dim, 0.0);
        dot_rows(query, keys, self.kv_dim, offset, self.scale, scores);
        softmax(scores);
        mixed.fill(0.0);
        mix_rows(scores, values, self.kv_dim, offset, mixed);
    }
}

/// Grouped-query self-attention for a batch of tokens, with reusable
/// scratch buffers. Rows named `*_rows` are laid out by weight row, the
/// rest token after token.
#[derive(Debug)]
pub(crate) struct Attention {
    heads: Heads,
    width: usize,
    query_rows: Vec<f32>,
    key_rows: Vec<f32>,
    value_rows: Vec<f32>,
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    mixed: Vec<f32>,
}

impl Attention {
    pub(crate) fn new(config: &LlamaConfig) -> Result<Self> {
        #[expect(
            clippy::cast_precision_loss,
            reason = "head dimensions are far below 2^24, where f32 is exact"
        )]
        let head_dim = config.head_dim as f32;
        let heads = Heads {
            head_dim: config.head_dim,
            group: config.head_count / config.head_count_kv,
            kv_dim: config.kv_dim()?,
            scale: head_dim.sqrt(),
            rope_freq_base: config.rope_freq_base,
            rope: config.variant.rope(),
        };
        Ok(Self {
            heads,
            width: config.embedding_length,
            query_rows: Vec::new(),
            key_rows: Vec::new(),
            value_rows: Vec::new(),
            query: Vec::new(),
            key: Vec::new(),
            value: Vec::new(),
            mixed: Vec::new(),
        })
    }

    /// Writes `attn_output(attention(input))` for the tokens of `input`, one
    /// row each, at positions `start..`, storing their keys and values in
    /// `cache` under `layer`. Each token attends only to the rows cached up
    /// to and including its own.
    pub(crate) fn forward(
        &mut self,
        block: &BlockWeights,
        cache: &mut KvCache,
        layer: usize,
        start: usize,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        let tokens = input.len() / self.width;
        self.project(block, input, tokens);
        let (width, kv_dim) = (self.width, self.heads.kv_dim);
        let rows = self
            .query
            .chunks_exact_mut(width)
            .zip(self.key.chunks_exact_mut(kv_dim))
            .zip(self.value.chunks_exact_mut(kv_dim));
        for (offset, ((query, key), value)) in rows.enumerate() {
            let position = start + offset;
            add_bias(query, block.attn_q_bias.as_deref());
            add_bias(key, block.attn_k_bias.as_deref());
            add_bias(value, block.attn_v_bias.as_deref());
            self.heads.rotate(query, key, position);
            cache.store(layer, position, key, value)?;
        }
        self.attend(cache.keys(layer), cache.values(layer), start);
        let jobs = &mut [(block.attn_output.weights(), &mut self.query_rows[..])];
        matmul_batch(&self.mixed, tokens, jobs);
        to_token_major(&self.query_rows, tokens, output);
        Ok(())
    }

    /// Mixes every (token, head) of the batch in parallel, all cached first.
    /// Token `t` sees only the rows up to position `start + t`, which keeps
    /// attention causal. Decode takes this path too: its heads gain more
    /// from the parallel region than the region costs.
    fn attend(&mut self, keys: &[f32], values: &[f32], start: usize) {
        let (heads, head_dim) = (self.heads, self.heads.head_dim);
        let head_count = self.width / head_dim;
        self.mixed.resize(self.query.len(), 0.0);
        let items = self
            .mixed
            .par_chunks_exact_mut(head_dim)
            .zip(self.query.par_chunks_exact(head_dim));
        items
            .enumerate()
            .for_each_init(Vec::new, |scores, (index, (mixed, query))| {
                let rows = (start + index / head_count + 1) * heads.kv_dim;
                let cache = (
                    keys.get(..rows).unwrap_or(keys),
                    values.get(..rows).unwrap_or(values),
                );
                heads.mix(index % head_count, query, cache, scores, mixed);
            });
    }

    /// Projects `input` to token-major queries, keys and values.
    fn project(&mut self, block: &BlockWeights, input: &[f32], tokens: usize) {
        let kv_len = self.heads.kv_dim * tokens;
        self.query_rows.resize(input.len(), 0.0);
        self.key_rows.resize(kv_len, 0.0);
        self.value_rows.resize(kv_len, 0.0);
        matmul_batch(
            input,
            tokens,
            &mut [
                (block.attn_q.weights(), &mut self.query_rows),
                (block.attn_k.weights(), &mut self.key_rows),
                (block.attn_v.weights(), &mut self.value_rows),
            ],
        );
        self.query.resize(input.len(), 0.0);
        self.key.resize(kv_len, 0.0);
        self.value.resize(kv_len, 0.0);
        to_token_major(&self.query_rows, tokens, &mut self.query);
        to_token_major(&self.key_rows, tokens, &mut self.key);
        to_token_major(&self.value_rows, tokens, &mut self.value);
    }
}

fn add_bias(projection: &mut [f32], bias: Option<&[f32]>) {
    if let Some(bias) = bias {
        add_assign(projection, bias);
    }
}

#[cfg(test)]
#[path = "attention_tests.rs"]
mod tests;
