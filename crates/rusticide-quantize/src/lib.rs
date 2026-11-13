// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Block quantization formats: `Q4_0`, `Q4_K`, `Q5_0`, `Q6_K`, `Q8_0` and
//! the `BitNet` ternary `I2_S`.

mod error;
mod format;
mod half_bits;
mod i2_s;
mod q4_0;
mod q4_k;
mod q5_0;
mod q6_k;
mod q8_0;

pub use error::QuantizeError;
pub use format::{BlockFormat, block_format, formats};
pub use half_bits::read_f16_le;
pub use i2_s::i2_s_scale;
pub use q4_0::Quantizer;
