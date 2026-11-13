// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_model::Matrix;

/// The weights of one transformer block. `attn_sub_norm` and `ffn_sub_norm`
/// are the optional norms before `attn_output` and `ffn_down`.
#[derive(Debug)]
pub struct BlockWeights {
    pub attn_norm: Vec<f32>,
    pub attn_q: Matrix,
    pub attn_k: Matrix,
    pub attn_v: Matrix,
    pub attn_q_bias: Option<Vec<f32>>,
    pub attn_k_bias: Option<Vec<f32>>,
    pub attn_v_bias: Option<Vec<f32>>,
    pub attn_output: Matrix,
    pub ffn_norm: Vec<f32>,
    pub ffn_gate: Matrix,
    pub ffn_up: Matrix,
    pub ffn_down: Matrix,
    pub attn_sub_norm: Option<Vec<f32>>,
    pub ffn_sub_norm: Option<Vec<f32>>,
}

/// Every weight of a decoder; `output` is `None` when tied to `token_embd`.
#[derive(Debug)]
pub struct DecoderWeights {
    pub token_embd: Matrix,
    pub blocks: Vec<BlockWeights>,
    pub output_norm: Vec<f32>,
    pub output: Option<Matrix>,
}
