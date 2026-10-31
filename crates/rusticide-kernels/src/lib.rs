// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared slice-level `f32` kernels.
//!
//! Mismatched slice lengths never panic: kernels iterate with `zip`, so the
//! shorter slice bounds the work. Callers validate lengths.

mod dot;
mod elementwise;
mod norm;
mod rope;
#[cfg(test)]
mod test_support;

pub use dot::{dot, matvec};
pub use elementwise::{add_assign, mul_assign, silu};
pub use norm::{rms_norm, softmax};
pub use rope::{rope_adjacent, rope_half_split};

#[expect(
    clippy::cast_precision_loss,
    reason = "counts beyond 2^24 lose low bits, an accepted f32 rounding"
)]
const fn lossy(count: usize) -> f32 {
    count as f32
}
