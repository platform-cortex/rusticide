// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rusticide SIMD: NEON kernels behind safe entry points.
//!
//! The second crate in the workspace permitted to use `unsafe`. See
//! `docs/UNSAFE.md` for the policy and this crate's `README.md` for why.
//! The `neon` module exists only when the build target guarantees NEON
//! and `dotprod`; [`AVAILABLE`] says whether it was compiled in.

/// True exactly when the `neon` module is compiled in.
pub const AVAILABLE: bool = cfg!(all(target_arch = "aarch64", target_feature = "dotprod"));

#[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
pub mod neon {
    #[cfg(not(miri))]
    mod blocks;
    pub mod dot;
    pub mod group;
    pub mod q4_k;
    pub mod q5_0;
    pub mod q6_k;
    pub mod q8_0;
    #[cfg(any(miri, test))]
    mod scalar;
    #[cfg(test)]
    mod test_support;
    #[cfg(not(miri))]
    mod vector;
}

#[cfg(test)]
mod tests {
    use super::AVAILABLE;

    #[test]
    fn available_matches_the_build_target() {
        assert_eq!(
            AVAILABLE,
            cfg!(all(target_arch = "aarch64", target_feature = "dotprod"))
        );
    }
}
