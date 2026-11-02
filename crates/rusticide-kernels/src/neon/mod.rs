// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
mod group_checks;
mod group_dot;
mod group_layout;
mod q4_k;
mod q4_k_dot;
mod q5_0;
mod q5_0_dot;
mod q6_k;
mod q6_k_dot;
mod q8_0;
mod q8_0_dot;

use std::sync::OnceLock;

use rusticide_simd::Dotprod;

use crate::adapter::Adapter;
use crate::portable;
use crate::quant_kernel::QuantKernel;

/// The portable kernels in the same order, all but `Q4_0` replaced by the
/// NEON ones.
static QUANTIZED: [QuantKernel; 5] = [
    portable::Q4_0,
    q5_0_dot::KERNEL,
    q4_k_dot::KERNEL,
    q6_k_dot::KERNEL,
    q8_0_dot::KERNEL,
];

/// The portable kernels, with NEON kernels in place of some of them.
pub(crate) static ADAPTER: Adapter = Adapter {
    name: "neon",
    quantized: &QUANTIZED,
    ..portable::ADAPTER
};

/// The proof the kernel wrappers hand their SIMD kernels, stored by
/// adapter detection; while it is unset they compute with the portable
/// kernels.
static TOKEN: OnceLock<Dotprod> = OnceLock::new();

/// Stores `proof` for the kernel wrappers and returns the adapter.
pub(crate) fn armed(proof: Dotprod) -> &'static Adapter {
    TOKEN.get_or_init(|| proof);
    &ADAPTER
}

/// The stored proof, `None` before detection found `dotprod`.
fn proof() -> Option<Dotprod> {
    TOKEN.get().copied()
}
