// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_model::{Model, ModelSource, Result};
use rusticide_transformer::Decoder;

use crate::config::LlamaConfig;
use crate::variant::Variant;
use crate::weights::load_weights;

/// Reads the configuration and weights of `source` into a decoder.
fn decoder(source: &dyn ModelSource, variant: Variant) -> Result<Decoder> {
    let config = LlamaConfig::from_source(source, variant)?;
    let weights = load_weights(source, &config)?;
    Decoder::new(config.decoder(), weights)
}

/// The registry [`rusticide_model::Loader`] for the `llama` architecture.
///
/// # Errors
///
/// Any error from [`LlamaConfig::from_source`], from loading a weight or
/// from [`Decoder::new`].
pub fn load(source: &dyn ModelSource) -> Result<Box<dyn Model>> {
    Ok(Box::new(decoder(source, Variant::Llama)?))
}

/// The registry [`rusticide_model::Loader`] for the `qwen2` architecture.
///
/// # Errors
///
/// As [`load`].
pub fn load_qwen2(source: &dyn ModelSource) -> Result<Box<dyn Model>> {
    Ok(Box::new(decoder(source, Variant::Qwen2)?))
}

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;
