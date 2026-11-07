// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The kernel on AVX2. A weight is its nibble with `0xf0` set unless its
//! fifth bit is: the nibble plus 16 times the bit, minus 16, as an i8, as
//! the NEON body builds it. The eight rows' fifth-bit words fill one
//! vector, a lane per row; a byte shuffle and a bit test give each byte
//! its weight's bit.

use core::arch::x86_64::{
    __m256i, _mm256_and_si256, _mm256_andnot_si256, _mm256_cmpeq_epi8, _mm256_or_si256,
    _mm256_set1_epi8, _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_srli_epi16,
};

use super::UNIT;
use crate::avx2::blocks::{self, Unpacked};
use crate::group::q5_0::{D, QH, QS};
use crate::group::{Q8Blocks, ROWS};

// `unpack` hard-codes these offsets in its pattern.
const _: () = assert!(D == 0 && QH == 16 && QS == 48 && UNIT == QS + 4 * 32);

/// For fifth-bit byte `j`: each four-byte lane of a 128-bit half repeats
/// byte `j` of its row's word, as `vpshufb` indexes within each half.
const BYTE: [[u8; 16]; 4] = [
    [0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12],
    [1, 1, 1, 1, 5, 5, 5, 5, 9, 9, 9, 9, 13, 13, 13, 13],
    [2, 2, 2, 2, 6, 6, 6, 6, 10, 10, 10, 10, 14, 14, 14, 14],
    [3, 3, 3, 3, 7, 7, 7, 7, 11, 11, 11, 11, 15, 15, 15, 15],
];

/// The bit of a lane's weight in its fifth-bit byte, for an even and an
/// odd group of four weights.
const BIT: [[u8; 16]; 2] = [
    [1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8],
    [
        16, 32, 64, 128, 16, 32, 64, 128, 16, 32, 64, 128, 16, 32, 64, 128,
    ],
];

/// `half` in both 128-bit halves of a vector.
fn both_halves(half: [u8; 16]) -> __m256i {
    bytemuck::cast([half, half])
}

/// The weights of group `group` (elements `4 * group..`) of every row,
/// from their nibbles and the rows' fifth-bit words.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn with_fifth(nibbles: __m256i, words: __m256i, group: usize) -> __m256i {
    let bytes = _mm256_shuffle_epi8(words, both_halves(BYTE[group / 2]));
    let bit = both_halves(BIT[group % 2]);
    let set = _mm256_cmpeq_epi8(_mm256_and_si256(bytes, bit), bit);
    _mm256_or_si256(
        nibbles,
        _mm256_andnot_si256(set, _mm256_set1_epi8(0xf0_u8.cast_signed())),
    )
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn unpack(unit: &[u8; UNIT]) -> Unpacked {
    let lanes: &[[u8; 16]; 11] = bytemuck::cast_ref(unit);
    let [d, low_words, high_words, packed @ ..] = lanes;
    let words: __m256i = bytemuck::cast([*low_words, *high_words]);
    let packed: &[[u8; 32]; 4] = bytemuck::cast_ref(packed);
    let mask = _mm256_set1_epi8(0x0f);
    let mut weights = [_mm256_setzero_si256(); 8];
    for (group, &bytes) in packed.iter().enumerate() {
        let bytes: __m256i = bytemuck::cast(bytes);
        let low = _mm256_and_si256(bytes, mask);
        let high = _mm256_and_si256(_mm256_srli_epi16::<4>(bytes), mask);
        weights[group] = with_fifth(low, words, group);
        weights[group + 4] = with_fifth(high, words, group + 4);
    }
    (*d, weights)
}

#[target_feature(enable = "avx2,fma")]
pub(super) fn group(units: &[[u8; UNIT]], inputs: &[Q8Blocks<'_>], out: &mut [[f32; ROWS]]) {
    // A `#[target_feature]` fn does not implement `Fn`, so `unpack` cannot
    // be passed by name; the closure, built where the features are enabled,
    // can.
    blocks::group(units, inputs, out, |unit| unpack(unit));
}
