// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_model::{Matrix, ModelSource, Result, load_vector};

use crate::config::DecoderConfig;
use crate::family::Family;
use crate::weights::{BlockWeights, DecoderWeights};

impl DecoderWeights {
    /// Loads every weight of `source` by its tensor name; `output` is
    /// `None` when `output.weight` is absent, tied to `token_embd`.
    ///
    /// # Errors
    ///
    /// `MissingTensor` or `ShapeMismatch` for any absent or mis-shaped
    /// weight, and `InvalidConfig` when the configuration's sizes overflow.
    pub fn from_source(
        source: &dyn ModelSource,
        config: &DecoderConfig,
        family: &Family,
    ) -> Result<Self> {
        let embedding = config.embedding_length;
        let mut blocks = Vec::new();
        for index in 0..config.block_count {
            blocks.push(load_block(source, config, family, index)?);
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

/// Loads block `index` by its tensor names.
fn load_block(
    source: &dyn ModelSource,
    config: &DecoderConfig,
    family: &Family,
    index: usize,
) -> Result<BlockWeights> {
    let embedding = config.embedding_length;
    let feed_forward = config.feed_forward_length;
    let kv_dim = config.kv_dim()?;
    let name = |part: &str| format!("blk.{index}.{part}.weight");
    let matrix =
        |part: &str, rows: usize, cols: usize| Matrix::load(source, &name(part), rows, cols);
    let bias = |part: &str, len: usize| {
        let name = format!("blk.{index}.{part}.bias");
        if source.tensor_shape(&name).is_some() || family.requires_attention_bias {
            load_vector(source, &name, len).map(Some)
        } else {
            Ok(None)
        }
    };
    let sub_norm = |part: &str, len: usize| {
        if family.sub_norms {
            load_vector(source, &name(part), len).map(Some)
        } else {
            Ok(None)
        }
    };
    Ok(BlockWeights {
        attn_norm: load_vector(source, &name("attn_norm"), embedding)?,
        attn_q: matrix("attn_q", embedding, embedding)?,
        attn_k: matrix("attn_k", kv_dim, embedding)?,
        attn_v: matrix("attn_v", kv_dim, embedding)?,
        attn_q_bias: bias("attn_q", embedding)?,
        attn_k_bias: bias("attn_k", kv_dim)?,
        attn_v_bias: bias("attn_v", kv_dim)?,
        attn_output: matrix("attn_output", embedding, embedding)?,
        ffn_norm: load_vector(source, &name("ffn_norm"), embedding)?,
        ffn_gate: matrix("ffn_gate", feed_forward, embedding)?,
        ffn_up: matrix("ffn_up", feed_forward, embedding)?,
        ffn_down: matrix("ffn_down", embedding, feed_forward)?,
        attn_sub_norm: sub_norm("attn_sub_norm", embedding)?,
        ffn_sub_norm: sub_norm("ffn_sub_norm", feed_forward)?,
    })
}
