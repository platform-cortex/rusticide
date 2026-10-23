// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Llama-family architecture: configuration and weights.

mod config;
mod weights;

#[cfg(any(test, feature = "testing"))]
pub mod testing;

pub use config::LlamaConfig;
pub use weights::{BlockWeights, LlamaWeights};
