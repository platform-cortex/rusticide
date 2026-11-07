// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `f32` dot kernels of [`crate::float`], compiled with `avx2,fma`.
//!
//! `x86_64`'s baseline has no FMA, so the portable `mul_add` is a call to
//! the C library's `fmaf`; with `fma` enabled it is one instruction. Both
//! round once, so the results are bit-identical for every non-NaN input
//! (NaN payloads may differ, and Rust does not specify them).

use crate::Avx2;

#[cfg(miri)]
use crate::float as body;

/// [`crate::float::dot`] with `fma`.
#[inline]
#[must_use]
#[cfg_attr(miri, allow(unused_unsafe))]
pub fn dot(_token: Avx2, lhs: &[f32], rhs: &[f32]) -> f32 {
    // SAFETY: an `Avx2` exists only after `Avx2::detect` found avx2 and
    // fma on this CPU, so the CPU running this call has every feature the
    // body enables.
    unsafe { body::dot(lhs, rhs) }
}

/// [`crate::float::dot_rows`] with `fma`.
#[inline]
#[cfg_attr(miri, allow(unused_unsafe))]
pub fn dot_rows(
    _token: Avx2,
    query: &[f32],
    rows: &[f32],
    row_stride: usize,
    offset: usize,
    scale: f32,
    out: &mut [f32],
) {
    // SAFETY: an `Avx2` exists only after `Avx2::detect` found avx2 and
    // fma on this CPU, so the CPU running this call has every feature the
    // body enables.
    unsafe { body::dot_rows(query, rows, row_stride, offset, scale, out) }
}

/// [`crate::float::mix_rows`] with `fma`.
#[inline]
#[cfg_attr(miri, allow(unused_unsafe))]
pub fn mix_rows(
    _token: Avx2,
    weights: &[f32],
    rows: &[f32],
    row_stride: usize,
    offset: usize,
    out: &mut [f32],
) {
    // SAFETY: an `Avx2` exists only after `Avx2::detect` found avx2 and
    // fma on this CPU, so the CPU running this call has every feature the
    // body enables.
    unsafe { body::mix_rows(weights, rows, row_stride, offset, out) }
}

/// The portable bodies, inlined where `avx2,fma` is enabled.
#[cfg(not(miri))]
mod body {
    use crate::float;

    #[target_feature(enable = "avx2,fma")]
    pub(super) fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
        float::dot(lhs, rhs)
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) fn dot_rows(
        query: &[f32],
        rows: &[f32],
        row_stride: usize,
        offset: usize,
        scale: f32,
        out: &mut [f32],
    ) {
        float::dot_rows(query, rows, row_stride, offset, scale, out);
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) fn mix_rows(
        weights: &[f32],
        rows: &[f32],
        row_stride: usize,
        offset: usize,
        out: &mut [f32],
    ) {
        float::mix_rows(weights, rows, row_stride, offset, out);
    }
}

#[cfg(test)]
mod tests {
    use super::{dot, dot_rows, mix_rows};
    use crate::float;
    use crate::test_support::token;

    fn values(rng: &mut fastrand::Rng, len: usize) -> Vec<f32> {
        (0..len)
            .map(|_| rng.f32().mul_add(2.0, -1.0) * 1e3)
            .collect()
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    #[test]
    fn dot_is_bit_identical_to_portable() {
        let mut rng = fastrand::Rng::with_seed(0x00f3_2d07);
        for len in 0..70 {
            let (lhs, rhs) = (values(&mut rng, len), values(&mut rng, len + len % 3));
            let want = float::dot(&lhs, &rhs);
            assert_eq!(dot(token(), &lhs, &rhs).to_bits(), want.to_bits(), "{len}");
        }
    }

    #[test]
    fn dot_rows_is_bit_identical_to_portable() {
        let mut rng = fastrand::Rng::with_seed(0x00f3_2d08);
        for (query_len, stride, offset) in [(64, 256, 64), (19, 40, 3), (8, 8, 0), (5, 4, 9)] {
            let query = values(&mut rng, query_len);
            let rows = values(&mut rng, stride * 6);
            let (mut got, mut want) = ([0.0; 7], [0.0; 7]);
            dot_rows(token(), &query, &rows, stride, offset, 8.0, &mut got);
            float::dot_rows(&query, &rows, stride, offset, 8.0, &mut want);
            assert_eq!(bits(&got), bits(&want), "{query_len} {stride} {offset}");
        }
    }

    #[test]
    fn mix_rows_is_bit_identical_to_portable() {
        let mut rng = fastrand::Rng::with_seed(0x00f3_2d09);
        for (out_len, stride, offset) in [(64, 256, 128), (19, 40, 3), (8, 8, 0), (5, 4, 9)] {
            let weights = values(&mut rng, 6);
            let rows = values(&mut rng, stride * 5);
            let mut got = values(&mut rng, out_len);
            let mut want = got.clone();
            mix_rows(token(), &weights, &rows, stride, offset, &mut got);
            float::mix_rows(&weights, &rows, stride, offset, &mut want);
            assert_eq!(bits(&got), bits(&want), "{out_len} {stride} {offset}");
        }
    }
}
