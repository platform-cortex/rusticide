// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Decoder-only transformer building blocks.

mod batch;
mod layout;
mod norm;
mod rope;

pub use batch::{PARALLEL_MIN_TOKENS, for_each_row, zip_rows};
pub use layout::to_token_major;
pub use norm::{normalize, normalize_rows};
pub use rope::RopeStyle;
