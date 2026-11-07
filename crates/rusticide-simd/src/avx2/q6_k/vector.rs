// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The kernel on AVX2. Per unit, each 32-element run's weights are
//! unpacked once into eight vectors of signed weights `w = q - 32` and
//! eight of their magnitudes, and multiplied with every input of the
//! chunk.
//!
//! `vpmaddubsw` takes unsigned bytes first, so it multiplies `|w|` by the
//! activation with `w`'s sign (`vpsignb`), which is exact because the
//! views refuse the quant `-128`. With `|w| <= 32` and `|a| <= 127` a pair
//! sum is at most `8128` in magnitude, and the four vectors of a
//! 16-element half-run at most `32512`: no `i16` lane saturates or wraps.
//! `vpmaddwd` by the half-run's scale then leaves exact `i32` sums, at
//! most `2^27` in magnitude per block, as in the portable kernel.

use core::arch::x86_64::{
    __m256, __m256i, _mm_cvtsi64_si128, _mm256_abs_epi8, _mm256_add_epi16, _mm256_add_epi32,
    _mm256_add_ps, _mm256_and_si256, _mm256_blend_epi16, _mm256_cvtepi8_epi32, _mm256_cvtepi32_ps,
    _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_mul_ps, _mm256_or_si256, _mm256_set1_epi8,
    _mm256_set1_epi32, _mm256_set1_ps, _mm256_setzero_si256, _mm256_sign_epi8, _mm256_slli_epi16,
    _mm256_slli_epi32, _mm256_srli_epi16, _mm256_sub_epi8,
};

use super::UNIT;
use crate::avx2::vector::halves;
use crate::group::q6_k::{D, QH, QL, SCALES};
use crate::group::{Q8Superblocks, ROWS, TOKENS};

/// Eight 32-byte vectors: the same four bytes of 32 per row, for all rows.
type Lanes = [[u8; 32]; 8];

// `split` hard-codes these offsets in its patterns.
const _: () = assert!(QL == 0 && QH == 32 * 32 && SCALES == QH + 32 * 16 && D == SCALES + 128);
const _: () = assert!(UNIT == D + 16);

/// A unit as vectors: the low bits of the four quarters of each half (the
/// first and second 32 bytes of each half's 64), the high bits of the two
/// halves, the 16 scales (each in both `i16` halves of its row's lane, as
/// `vpmaddwd` reads it) and `d`.
struct Unit<'a> {
    low: [&'a Lanes; 4],
    high: [&'a Lanes; 2],
    scales: [__m256i; 16],
    d: __m256,
}

/// One run's weights: their magnitudes and the weights themselves.
struct Weights {
    magnitudes: [__m256i; 8],
    signs: [__m256i; 8],
}

/// Eight signed bytes, one per row, in both `i16` halves of `i32` lanes.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn scale_pairs(bytes: [u8; 8]) -> __m256i {
    let wide = _mm256_cvtepi8_epi32(_mm_cvtsi64_si128(i64::from_le_bytes(bytes)));
    _mm256_blend_epi16::<0b1010_1010>(wide, _mm256_slli_epi32::<16>(wide))
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn split(unit: &[u8; UNIT]) -> Unit<'_> {
    let lanes: &[[u8; 16]; 105] = bytemuck::cast_ref(unit);
    let [eights @ .., d] = lanes;
    let eights: &[[u8; 128]; 13] = bytemuck::cast_ref(eights);
    let [bits @ .., scales] = eights;
    let [l0, l1, l2, l3, h0, h1]: &[Lanes; 6] = bytemuck::cast_ref(bits);
    let scales: &[[u8; 8]; 16] = bytemuck::cast_ref(scales);
    Unit {
        low: [l0, l1, l2, l3],
        high: [h0, h1],
        scales: scales.map(|bytes| scale_pairs(bytes)),
        d: halves(*d),
    }
}

/// Run `run` (`0..4`) of a half: `low` holds its nibbles, `high` its two
/// high bits at `2 * run`.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn unpack(low: &Lanes, high: &Lanes, run: usize) -> Weights {
    let (nibble_mask, top_mask) = (_mm256_set1_epi8(0x0f), _mm256_set1_epi8(0x30));
    let offset = _mm256_set1_epi8(32);
    let mut weights = Weights {
        magnitudes: [_mm256_setzero_si256(); 8],
        signs: [_mm256_setzero_si256(); 8],
    };
    for (index, (&low, &high)) in low.iter().zip(high).enumerate() {
        let (low, high): (__m256i, __m256i) = (bytemuck::cast(low), bytemuck::cast(high));
        let nibble = if run < 2 {
            low
        } else {
            _mm256_srli_epi16::<4>(low)
        };
        let top = match run {
            0 => _mm256_slli_epi16::<4>(high),
            1 => _mm256_slli_epi16::<2>(high),
            2 => high,
            _ => _mm256_srli_epi16::<2>(high),
        };
        let unsigned = _mm256_or_si256(
            _mm256_and_si256(nibble, nibble_mask),
            _mm256_and_si256(top, top_mask),
        );
        let signed = _mm256_sub_epi8(unsigned, offset);
        weights.signs[index] = signed;
        weights.magnitudes[index] = _mm256_abs_epi8(signed);
    }
    weights
}

/// Adds one 32-element run's dot products, each half-run's times its
/// scale, to `sum`. Weight vector `i` holds elements `4i..4i + 4` of every
/// row.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn run(sum: __m256i, weights: &Weights, quants: &[[i8; 4]; 8], scales: [__m256i; 2]) -> __m256i {
    let mut sum = sum;
    let (quants, _) = quants.as_chunks::<4>();
    let (magnitudes, _) = weights.magnitudes.as_chunks::<4>();
    let (signs, _) = weights.signs.as_chunks::<4>();
    let halves = quants.iter().zip(magnitudes).zip(signs);
    for (((quants, magnitudes), signs), scale) in halves.zip(scales) {
        let mut dots = _mm256_setzero_si256();
        for ((&quad, &magnitude), &sign) in quants.iter().zip(magnitudes).zip(signs) {
            let quad = _mm256_set1_epi32(bytemuck::cast(quad));
            let signed = _mm256_sign_epi8(quad, sign);
            dots = _mm256_add_epi16(dots, _mm256_maddubs_epi16(magnitude, signed));
        }
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(dots, scale));
    }
    sum
}

/// One unit's integer sums for every input of the chunk.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn sums<const N: usize>(
    unit: &Unit<'_>,
    block: usize,
    inputs: &[Q8Superblocks<'_>],
) -> [__m256i; N] {
    let mut sums = [_mm256_setzero_si256(); N];
    for (half, high) in unit.high.iter().enumerate() {
        for quarter in 0..4 {
            let weights = unpack(unit.low[2 * half + quarter % 2], high, quarter);
            let index = 4 * half + quarter;
            let pair = [unit.scales[2 * index], unit.scales[2 * index + 1]];
            for (sum, input) in sums.iter_mut().zip(inputs) {
                if let Some(quants) = input.quants.get(block) {
                    let runs: &[[[i8; 4]; 8]; 8] = bytemuck::cast_ref(quants);
                    *sum = run(*sum, &weights, &runs[index], pair);
                }
            }
        }
    }
    sums
}

/// `acc + d * d8 * sum` per row, as the portable kernel rounds it: two
/// multiplies and an add, none fused.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn combine(acc: __m256, sum: __m256i, d: __m256, d8: f32) -> __m256 {
    let term = _mm256_mul_ps(
        _mm256_mul_ps(d, _mm256_set1_ps(d8)),
        _mm256_cvtepi32_ps(sum),
    );
    _mm256_add_ps(acc, term)
}

/// [`group`] for at most `N` inputs.
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
fn chunk<const N: usize>(
    units: &[[u8; UNIT]],
    inputs: &[Q8Superblocks<'_>],
    out: &mut [[f32; ROWS]],
) {
    let mut acc = [_mm256_set1_ps(-0.0); N];
    for (block, unit) in units.iter().enumerate() {
        let unit = split(unit);
        let sums = sums::<N>(&unit, block, inputs);
        for ((acc, &sum), input) in acc.iter_mut().zip(&sums).zip(inputs) {
            if let Some(scale) = input.scales.get(block) {
                *acc = combine(*acc, sum, unit.d, scale[0]);
            }
        }
    }
    for (rows, acc) in out.iter_mut().zip(acc) {
        *rows = bytemuck::cast(acc);
    }
}

#[target_feature(enable = "avx2,fma")]
pub(super) fn group(units: &[[u8; UNIT]], inputs: &[Q8Superblocks<'_>], out: &mut [[f32; ROWS]]) {
    // One input, as in decoding, gets a body whose sums all stay in registers.
    // It reads a copy of the view: through the caller's slice, the loop
    // would reload the view's slices on every unit.
    if let [input] = inputs {
        return chunk::<1>(units, &[*input], out);
    }
    for (inputs, out) in inputs.chunks(TOKENS).zip(out.chunks_mut(TOKENS)) {
        chunk::<TOKENS>(units, inputs, out);
    }
}
