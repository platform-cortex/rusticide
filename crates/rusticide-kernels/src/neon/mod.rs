// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::OnceLock;

use rusticide_simd::Dotprod;

use crate::adapter::Adapter;
use crate::group_dot::group_kernel;
use crate::portable;
use crate::quant_kernel::QuantKernel;

group_kernel!(
    q4_k_dot: Q4_K,
    crate::q4_k_layout,
    rusticide_simd::neon::q4_k::q4_k_group,
    superblocks
);
group_kernel!(
    q5_0_dot: Q5_0,
    crate::q5_0_layout,
    rusticide_simd::neon::q5_0::q5_0_group,
    blocks
);
group_kernel!(
    q6_k_dot: Q6_K,
    crate::q6_k_layout,
    rusticide_simd::neon::q6_k::q6_k_group,
    superblocks
);
group_kernel!(
    q8_0_dot: Q8_0,
    crate::q8_0_layout,
    rusticide_simd::neon::q8_0::q8_0_group,
    blocks
);

/// The proof token the NEON kernels take.
type Proof = Dotprod;

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
pub(crate) fn proof() -> Option<Proof> {
    TOKEN.get().copied()
}
