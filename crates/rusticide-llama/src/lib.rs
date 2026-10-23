// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Llama-family architecture: configuration, weights and the forward pass.

mod attention;
mod config;
mod feed_forward;
mod model;
mod weights;

#[cfg(any(test, feature = "testing"))]
pub mod testing;

#[cfg(test)]
mod test_support;

pub use config::LlamaConfig;
pub use model::{Llama, load};
pub use weights::{BlockWeights, LlamaWeights};
