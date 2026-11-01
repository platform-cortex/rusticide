// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared slice-level `f32` kernels.
//!
//! Mismatched slice lengths never panic: kernels iterate with `zip`, so the
//! shorter slice bounds the work. Callers validate lengths.

mod adapter;
mod facade;
mod portable;
#[cfg(test)]
mod test_support;

pub use adapter::{Adapter, active};
pub use facade::{
    add_assign, dot, dot_rows, mul_assign, rms_norm, rope_adjacent, rope_half_split, silu_assign,
    softmax,
};

#[expect(
    clippy::cast_precision_loss,
    reason = "counts beyond 2^24 lose low bits, an accepted f32 rounding"
)]
const fn lossy(count: usize) -> f32 {
    count as f32
}
