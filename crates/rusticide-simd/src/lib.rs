// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rusticide SIMD: NEON kernels behind safe entry points.
//!
//! The second crate in the workspace permitted to use `unsafe`. See
//! `docs/UNSAFE.md` for the policy and this crate's `README.md` for why.
//! The `neon` module compiles on every `aarch64` target; each of its
//! kernels takes a `Dotprod` token, which only run-time detection makes.
//! On `x86_64` the `Avx2` token is proved the same way.

pub mod group;
mod token;

#[cfg(target_arch = "x86_64")]
pub use token::Avx2;
#[cfg(target_arch = "aarch64")]
pub use token::Dotprod;

#[cfg(target_arch = "aarch64")]
pub mod neon {
    #[cfg(not(miri))]
    mod blocks;
    pub mod dot;
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
