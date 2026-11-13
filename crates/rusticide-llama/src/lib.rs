// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Llama-family architecture: configuration and weights by tensor name,
//! loaded into a [`rusticide_transformer::Decoder`].

mod config;
mod model;
mod registry;
mod variant;
mod weights;

#[cfg(any(test, feature = "testing"))]
pub mod testing;

pub use config::LlamaConfig;
pub use model::{load, load_qwen2};
pub use registry::registry;
pub use rusticide_transformer::{BlockWeights, RopeStyle};
pub use variant::Variant;
