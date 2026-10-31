// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Block quantization formats: `Q4_0`, `Q4_K`, `Q5_0`, `Q6_K` and `Q8_0`.

mod error;
mod format;
mod half_bits;
mod q4_0;
mod q4_k;
mod q5_0;
mod q6_k;
mod q8_0;

pub use error::QuantizeError;
pub use format::{BlockFormat, block_format, formats};
pub use q4_0::{Q4_0_BLOCK_BYTES, Q4_0_BLOCK_SIZE, Quantizer, bytes_for_q4_0};
pub use q4_k::{Q4_K_BLOCK_BYTES, Q4_K_BLOCK_SIZE, bytes_for_q4_k};
pub use q5_0::{Q5_0_BLOCK_BYTES, Q5_0_BLOCK_SIZE};
pub use q6_k::{Q6_K_BLOCK_BYTES, Q6_K_BLOCK_SIZE, bytes_for_q6_k};
pub use q8_0::{Q8_0_BLOCK_BYTES, Q8_0_BLOCK_SIZE};
