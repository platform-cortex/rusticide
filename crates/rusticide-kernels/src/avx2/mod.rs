// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

mod q4_k_dot;

use std::sync::OnceLock;

use rusticide_simd::Avx2;

use crate::adapter::Adapter;
use crate::portable;
use crate::quant_kernel::QuantKernel;

/// The portable kernels in the same order, `Q4_K` replaced by the AVX2
/// one.
static QUANTIZED: [QuantKernel; 5] = [
    portable::Q4_0,
    portable::Q5_0,
    q4_k_dot::KERNEL,
    portable::Q6_K,
    portable::Q8_0,
];

/// The portable kernels, with AVX2 kernels in place of some of them.
pub(crate) static ADAPTER: Adapter = Adapter {
    name: "avx2",
    quantized: &QUANTIZED,
    ..portable::ADAPTER
};

/// The proof the kernel wrappers hand their SIMD kernels, stored by
/// adapter detection; while it is unset they compute with the portable
/// kernels.
static TOKEN: OnceLock<Avx2> = OnceLock::new();

/// Stores `proof` for the kernel wrappers and returns the adapter.
pub(crate) fn armed(proof: Avx2) -> &'static Adapter {
    TOKEN.get_or_init(|| proof);
    &ADAPTER
}

/// The stored proof, `None` before detection found AVX2 and FMA.
pub(crate) fn proof() -> Option<Avx2> {
    TOKEN.get().copied()
}

#[cfg(test)]
mod tests {
    use super::{ADAPTER, Adapter, Avx2, armed, proof};

    #[test]
    fn arming_stores_the_detected_proof() {
        let detected = Avx2::detect();
        let adapter = detected.map(armed);
        assert_eq!(proof().is_some(), detected.is_some());
        assert_eq!(adapter.map(Adapter::name), detected.map(|_| "avx2"));
        assert_eq!(ADAPTER.name(), "avx2");
    }
}
