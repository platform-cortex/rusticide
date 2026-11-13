// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Llama-family architecture: configuration, weights and the forward pass.

mod config;
mod model;
mod registry;
mod variant;
mod weights;

#[cfg(any(test, feature = "testing"))]
pub mod testing;

#[cfg(test)]
mod test_support;

pub use config::LlamaConfig;
pub use model::{Llama, load, load_qwen2};
pub use registry::registry;
pub use rusticide_transformer::{BlockWeights, RopeStyle};
pub use variant::Variant;
pub use weights::LlamaWeights;
