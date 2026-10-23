// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_model::{Matrix, ModelSource, Result, load_vector};

use crate::config::LlamaConfig;

/// The weights of one transformer block.
#[derive(Debug)]
pub struct BlockWeights {
    pub attn_norm: Vec<f32>,
    pub attn_q: Matrix,
    pub attn_k: Matrix,
    pub attn_v: Matrix,
    pub attn_output: Matrix,
    pub ffn_norm: Vec<f32>,
    pub ffn_gate: Matrix,
    pub ffn_up: Matrix,
    pub ffn_down: Matrix,
}

/// Every weight of a llama model; `output` is `None` when tied to `token_embd`.
#[derive(Debug)]
pub struct LlamaWeights {
    pub token_embd: Matrix,
    pub blocks: Vec<BlockWeights>,
    pub output_norm: Vec<f32>,
    pub output: Option<Matrix>,
}

impl LlamaWeights {
    /// # Errors
    ///
    /// `MissingTensor` or `ShapeMismatch` for any absent or mis-shaped weight,
    /// and `InvalidConfig` when the configuration's sizes overflow.
    pub fn load(source: &dyn ModelSource, config: &LlamaConfig) -> Result<Self> {
        let embedding = config.embedding_length;
        let mut blocks = Vec::new();
        for index in 0..config.block_count {
            blocks.push(BlockWeights::load(source, config, index)?);
        }
        let output = if source.tensor_shape("output.weight").is_some() {
            Some(Matrix::load(
                source,
                "output.weight",
                config.vocab_size,
                embedding,
            )?)
        } else {
            None
        };
        Ok(Self {
            token_embd: Matrix::load(source, "token_embd.weight", config.vocab_size, embedding)?,
            blocks,
            output_norm: load_vector(source, "output_norm.weight", embedding)?,
            output,
        })
    }
}

impl BlockWeights {
    fn load(source: &dyn ModelSource, config: &LlamaConfig, index: usize) -> Result<Self> {
        let embedding = config.embedding_length;
        let feed_forward = config.feed_forward_length;
        let kv_dim = config.kv_dim()?;
        let name = |part: &str| format!("blk.{index}.{part}.weight");
        let matrix =
            |part: &str, rows: usize, cols: usize| Matrix::load(source, &name(part), rows, cols);
        Ok(Self {
            attn_norm: load_vector(source, &name("attn_norm"), embedding)?,
            attn_q: matrix("attn_q", embedding, embedding)?,
            attn_k: matrix("attn_k", kv_dim, embedding)?,
            attn_v: matrix("attn_v", kv_dim, embedding)?,
            attn_output: matrix("attn_output", embedding, embedding)?,
            ffn_norm: load_vector(source, &name("ffn_norm"), embedding)?,
            ffn_gate: matrix("ffn_gate", feed_forward, embedding)?,
            ffn_up: matrix("ffn_up", feed_forward, embedding)?,
            ffn_down: matrix("ffn_down", embedding, feed_forward)?,
        })
    }
}

#[cfg(test)]
#[path = "weights_tests.rs"]
mod tests;
