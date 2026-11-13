// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_model::{ModelSource, Result};
use rusticide_transformer::{Activation, DecoderConfig, Family, RopeStyle};

/// The `general.architecture` of `BitNet` b1.58 models.
pub const ARCHITECTURE: &str = "bitnet-b1.58";

/// `BitNet` b1.58 on the shared decoder: half-split (NEOX) `RoPE`, a gated
/// squared-ReLU feed-forward and a sub-norm before `attn_output` and
/// `ffn_down` in every block.
pub(crate) const FAMILY: Family = Family {
    architecture: ARCHITECTURE,
    rope: RopeStyle::HalfSplit,
    activation: Activation::ReluSquared,
    requires_attention_bias: false,
    sub_norms: true,
};

/// Reads and validates the hyperparameters under the `bitnet-b1.58.` keys.
///
/// # Errors
///
/// As [`DecoderConfig::from_source`].
pub fn config(source: &dyn ModelSource) -> Result<DecoderConfig> {
    DecoderConfig::from_source(source, &FAMILY)
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
