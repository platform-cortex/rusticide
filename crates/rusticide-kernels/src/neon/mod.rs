// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

mod group_dot;
mod group_layout;
mod q4_k;
mod q4_k_dot;

use crate::adapter::Adapter;
use crate::portable;
use crate::quant_kernel::QuantKernel;

/// The portable kernels in the same order, `Q4_K` replaced by the NEON one.
static QUANTIZED: [QuantKernel; 5] = [
    portable::Q4_0,
    portable::Q5_0,
    q4_k_dot::KERNEL,
    portable::Q6_K,
    portable::Q8_0,
];

/// The portable kernels, with NEON kernels in place of some of them.
pub(crate) static ADAPTER: Adapter = Adapter {
    name: "neon",
    quantized: &QUANTIZED,
    ..portable::ADAPTER
};
