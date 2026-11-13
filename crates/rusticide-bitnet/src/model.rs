// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_model::{Model, ModelSource, Result};
use rusticide_transformer::{Decoder, DecoderWeights};

use crate::config::{FAMILY, config};

/// The registry [`rusticide_model::Loader`] for the `bitnet-b1.58`
/// architecture.
///
/// # Errors
///
/// Any error from [`config`], from loading a weight (both sub-norms are
/// required) or from [`Decoder::new`].
pub fn load(source: &dyn ModelSource) -> Result<Box<dyn Model>> {
    let config = config(source)?;
    let weights = DecoderWeights::from_source(source, &config, &FAMILY)?;
    Ok(Box::new(Decoder::new(config, weights)?))
}

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;
