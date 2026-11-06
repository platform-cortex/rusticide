// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! AVX2 helpers the group kernels share. A unit's eight rows fill the
//! eight lanes of one vector, so a row's sums never need a horizontal
//! reduction.

use core::arch::x86_64::{
    __m128i, __m256, _mm256_and_si256, _mm256_blendv_epi8, _mm256_castps_si256,
    _mm256_castsi256_ps, _mm256_cmpgt_epi32, _mm256_cvtepu16_epi32, _mm256_mul_ps, _mm256_or_si256,
    _mm256_set1_epi32, _mm256_set1_ps, _mm256_slli_epi32,
};

/// Eight little-endian halves, one per row, widened exactly as
/// `crate::scalar::f16_to_f32` does.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) fn halves(bytes: [u8; 16]) -> __m256 {
    let bits = _mm256_cvtepu16_epi32(bytemuck::cast::<[u8; 16], __m128i>(bytes));
    let magnitude = _mm256_and_si256(bits, _mm256_set1_epi32(0x7fff));
    let shifted = _mm256_slli_epi32::<13>(magnitude);
    let rebiased = _mm256_mul_ps(
        _mm256_castsi256_ps(shifted),
        _mm256_set1_ps(f32::from_bits(0x7780_0000)),
    );
    let special = _mm256_cmpgt_epi32(magnitude, _mm256_set1_epi32(0x7bff));
    let widened = _mm256_blendv_epi8(
        _mm256_castps_si256(rebiased),
        _mm256_or_si256(shifted, _mm256_set1_epi32(0x7f80_0000)),
        special,
    );
    let sign = _mm256_slli_epi32::<16>(_mm256_and_si256(bits, _mm256_set1_epi32(0x8000)));
    _mm256_castsi256_ps(_mm256_or_si256(widened, sign))
}
