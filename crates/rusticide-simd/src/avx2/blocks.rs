// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The AVX2 body both 32-element block formats share: per unit, the
//! format unpacks the eight rows' signed weights once into eight vectors,
//! vector `c` holding weights `4c..4c + 4` of every row, a lane per row,
//! and each input of the chunk adds `d * d8 * isum` per row.
//!
//! `vpmaddubsw` multiplies unsigned by signed bytes, so each weight `w`
//! goes in as `|w|` and its quant `a` as `vpsignb(a, w)`, `a` negated
//! where `w < 0` and zeroed where `w == 0`: `|w| * sign(a, w) = w * a`.
//! That needs `-a` to fit `i8`, which holds because the views refuse the
//! quant `-128`. `|w|` is at most 128 (`vpabsb` of `-128` is the byte
//! `0x80`, unsigned 128) and `|sign(a, w)|` at most 127, so a pair sums to
//! at most `2 * 128 * 127 = 32512` in magnitude: `vpmaddubsw` never
//! saturates. Each pair sum is widened by `vpmaddwd` with ones before it
//! is added, and a block's sum stays within `32 * 128 * 127 < 2^19`.

use core::arch::x86_64::{
    __m256, __m256i, _mm256_abs_epi8, _mm256_add_epi32, _mm256_add_ps, _mm256_cvtepi32_ps,
    _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_mul_ps, _mm256_set1_epi16, _mm256_set1_epi32,
    _mm256_set1_ps, _mm256_setzero_si256, _mm256_sign_epi8,
};

use crate::avx2::vector::halves;
use crate::group::{Q8Blocks, ROWS, TOKENS};

/// A unit's `d` bytes, one half per row, and its signed weights, vector
/// `c` holding weights `4c..4c + 4` of every row.
pub(crate) type Unpacked = ([u8; 16], [__m256i; 8]);

/// The eight rows' exact block sums against one input's quants.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn dot(magnitudes: &[__m256i; 8], weights: &[__m256i; 8], quants: &[i8; 32]) -> __m256i {
    let quads: &[[i8; 4]; 8] = bytemuck::cast_ref(quants);
    let ones = _mm256_set1_epi16(1);
    let mut sum = _mm256_setzero_si256();
    for ((&magnitude, &weight), &quad) in magnitudes.iter().zip(weights).zip(quads) {
        let signed = _mm256_sign_epi8(_mm256_set1_epi32(bytemuck::cast(quad)), weight);
        let pairs = _mm256_maddubs_epi16(magnitude, signed);
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(pairs, ones));
    }
    sum
}

/// `acc + d * d8 * sum` per row, as the portable kernels round it: two
/// multiplies and an add, none fused.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn combine(acc: __m256, sum: __m256i, d: __m256, d8: f32) -> __m256 {
    let scale = _mm256_mul_ps(d, _mm256_set1_ps(d8));
    _mm256_add_ps(acc, _mm256_mul_ps(scale, _mm256_cvtepi32_ps(sum)))
}

/// [`group`] for at most `N` inputs.
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
fn chunk<const UNIT: usize, const N: usize>(
    units: &[[u8; UNIT]],
    inputs: &[Q8Blocks<'_>],
    out: &mut [[f32; ROWS]],
    unpack: impl Fn(&[u8; UNIT]) -> Unpacked,
) {
    let mut acc = [_mm256_set1_ps(-0.0); N];
    for (block, unit) in units.iter().enumerate() {
        let (d, weights) = unpack(unit);
        let d = halves(d);
        let magnitudes = weights.map(|weight| _mm256_abs_epi8(weight));
        for (acc, input) in acc.iter_mut().zip(inputs) {
            if let (Some(quants), Some(&d8)) = (input.quants.get(block), input.scales.get(block)) {
                *acc = combine(*acc, dot(&magnitudes, &weights, quants), d, d8);
            }
        }
    }
    for (rows, acc) in out.iter_mut().zip(acc) {
        *rows = bytemuck::cast(acc);
    }
}

/// The products of a group's units with each input, every token computed
/// alone and in the same order.
#[target_feature(enable = "avx2,fma")]
pub(crate) fn group<const UNIT: usize>(
    units: &[[u8; UNIT]],
    inputs: &[Q8Blocks<'_>],
    out: &mut [[f32; ROWS]],
    unpack: impl Fn(&[u8; UNIT]) -> Unpacked + Copy,
) {
    // One input, as in decoding, gets a body whose sums all stay in registers.
    // It reads a copy of the view: through the caller's slice, the loop
    // would reload the view's slices on every unit.
    if let [input] = inputs {
        return chunk::<UNIT, 1>(units, &[*input], out, unpack);
    }
    for (inputs, out) in inputs.chunks(TOKENS).zip(out.chunks_mut(TOKENS)) {
        chunk::<UNIT, TOKENS>(units, inputs, out, unpack);
    }
}
