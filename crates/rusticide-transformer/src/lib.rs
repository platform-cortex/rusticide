// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Decoder-only transformer building blocks.

mod attention;
mod batch;
mod config;
mod feed_forward;
mod layout;
mod norm;
mod rope;
mod weights;

#[cfg(test)]
mod test_support;

pub use attention::Attention;
pub use batch::{PARALLEL_MIN_TOKENS, for_each_row, zip_rows};
pub use config::{Activation, DecoderConfig};
pub use feed_forward::FeedForward;
pub use layout::to_token_major;
pub use norm::{normalize, normalize_rows};
pub use rope::RopeStyle;
pub use weights::BlockWeights;
