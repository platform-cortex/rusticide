// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::config::Activation;
use crate::rope::RopeStyle;

/// What a model family fixes beyond the hyperparameters its metadata states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Family {
    /// The `general.architecture` value, which prefixes the family's keys.
    pub architecture: &'static str,
    pub rope: RopeStyle,
    pub activation: Activation,
    /// Whether every block must carry the query, key and value biases;
    /// otherwise each one is loaded only when present.
    pub requires_attention_bias: bool,
    /// Whether every block carries `attn_sub_norm` and `ffn_sub_norm`.
    pub sub_norms: bool,
}
