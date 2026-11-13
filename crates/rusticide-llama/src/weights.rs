// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_model::{ModelSource, Result};
use rusticide_transformer::DecoderWeights;

use crate::config::LlamaConfig;

/// Loads every weight of a llama model by its tensor name.
///
/// # Errors
///
/// As [`DecoderWeights::from_source`].
pub(crate) fn load_weights(
    source: &dyn ModelSource,
    config: &LlamaConfig,
) -> Result<DecoderWeights> {
    DecoderWeights::from_source(source, &config.decoder(), &config.variant.family())
}

#[cfg(test)]
#[path = "weights_tests.rs"]
mod tests;
