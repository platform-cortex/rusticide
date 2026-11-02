// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The kernel facade: slice-level `f32` kernels and the quantized row
//! kernels, routed through one adapter chosen per process, and the
//! scheduler that runs matrix products over them in parallel.
//!
//! Mismatched slice lengths never panic: kernels iterate with `zip`, so the
//! shorter slice bounds the work. Callers validate lengths.

mod adapter;
mod facade;
mod matmul;
#[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
mod neon;
mod portable;
mod q8_input;
mod quant_kernel;
#[cfg(any(test, feature = "test-kernels"))]
#[doc(hidden)]
pub mod test_support;

pub use adapter::{Adapter, active};
pub use facade::{
    add_assign, dot, dot_rows, mix_rows, mul_assign, rms_norm, rope_adjacent, rope_half_split,
    silu_assign, softmax,
};
pub use matmul::{Weights, matmul_batch, matvec_batch};
pub use q8_input::{Q8_BLOCK, Q8Input, Q8Kind};
pub use quant_kernel::QuantKernel;
