// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::OnceLock;

use rusticide_simd::Avx2;

use crate::adapter::Adapter;
use crate::portable;

/// The portable kernels under the `avx2` name until AVX2 kernels replace
/// some of them.
pub(crate) static ADAPTER: Adapter = Adapter {
    name: "avx2",
    ..portable::ADAPTER
};

/// The proof the kernel wrappers will hand their SIMD kernels, stored by
/// adapter detection.
static TOKEN: OnceLock<Avx2> = OnceLock::new();

/// Stores `proof` for the kernel wrappers and returns the adapter.
pub(crate) fn armed(proof: Avx2) -> &'static Adapter {
    TOKEN.get_or_init(|| proof);
    &ADAPTER
}

#[cfg(test)]
mod tests {
    use super::{ADAPTER, Adapter, Avx2, TOKEN, armed};

    #[test]
    fn arming_stores_the_detected_proof() {
        let detected = Avx2::detect();
        let adapter = detected.map(armed);
        assert_eq!(TOKEN.get().is_some(), detected.is_some());
        assert_eq!(adapter.map(Adapter::name), detected.map(|_| "avx2"));
        assert_eq!(ADAPTER.name(), "avx2");
    }
}
