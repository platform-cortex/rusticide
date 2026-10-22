// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Block quantization formats: `Q4_0`, `Q4_K` and `Q6_K`.

mod error;
mod half_bits;
mod q4_0;
mod q4_k;
mod q6_k;

pub use error::{QuantizationType, QuantizeError};
pub use q4_0::{Q4_0_BLOCK_BYTES, Q4_0_BLOCK_SIZE, Quantizer, bytes_for_q4_0};
pub use q4_k::{Q4_K_BLOCK_BYTES, Q4_K_BLOCK_SIZE, bytes_for_q4_k, dequantize_q4_k};
pub use q6_k::{Q6_K_BLOCK_BYTES, Q6_K_BLOCK_SIZE, bytes_for_q6_k, dequantize_q6_k};
