// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Token sampling strategies (greedy, top-k, top-p, temperature) over logits
//! held in `rusticide-core` tensors.

mod error;
mod filters;
mod logits;
mod params;
mod sampler;

pub use error::{Result, SamplingError};
pub use params::SamplingParams;
pub use sampler::Sampler;
