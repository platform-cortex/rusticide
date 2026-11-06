// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The kernel on AVX2. Per unit, each pair of 32-element runs' 256 weight
//! bytes are unpacked once into sixteen vectors and multiplied with every
//! input of the chunk.

use core::arch::x86_64::{
    __m256, __m256i, _mm_cvtsi64_si128, _mm256_add_epi16, _mm256_add_epi32, _mm256_add_ps,
    _mm256_and_si256, _mm256_cvtepi32_ps, _mm256_cvtepu8_epi32, _mm256_fmadd_ps, _mm256_madd_epi16,
    _mm256_maddubs_epi16, _mm256_mul_ps, _mm256_mullo_epi32, _mm256_or_si256, _mm256_set1_epi8,
    _mm256_set1_epi32, _mm256_set1_ps, _mm256_setzero_si256, _mm256_slli_epi32, _mm256_srli_epi16,
    _mm256_srli_epi32, _mm256_xor_ps,
};

use super::UNIT;
use crate::avx2::vector::halves;
use crate::group::q4_k::{D, DMIN, QS, SCALES};
use crate::group::{Q8Superblocks, ROWS, TOKENS};

// `lanes` and `quant_pairs` split a unit by these offsets.
const _: () = assert!(D == 0 && DMIN == 16 && SCALES == 32 && QS == 128 && UNIT == 1152);

/// What every input shares for one unit: the rows' float scales, their
/// 6-bit run scales (each in both `i16` halves of its row's lane, as
/// `vpmaddwd` reads it) and their run mins.
struct Header {
    d: __m256,
    dmin: __m256,
    scales: [__m256i; 8],
    mins: [__m256i; 8],
}

/// A unit as 16-byte lanes: `d`, `dmin`, six of packed scales, then the
/// quants.
fn lanes(unit: &[u8; UNIT]) -> &[[u8; 16]; 72] {
    bytemuck::cast_ref(unit)
}

/// The quants of a unit: four pairs of runs, eight 32-byte vectors each.
fn quant_pairs(unit: &[u8; UNIT]) -> &[[[u8; 32]; 8]; 4] {
    let [_, _, _, _, quants @ ..]: &[[u8; 32]; 36] = bytemuck::cast_ref(unit);
    bytemuck::cast_ref(quants)
}

/// Eight bytes, one per row, widened to `i32` lanes.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn rows_i32(bytes: [u8; 8]) -> __m256i {
    _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(i64::from_le_bytes(bytes)))
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn header(unit: &[u8; UNIT]) -> Header {
    let [d, dmin, s0, s1, s2, s3, s4, s5, ..] = *lanes(unit);
    let packed: [[u8; 8]; 12] = bytemuck::cast([s0, s1, s2, s3, s4, s5]);
    let (six, four) = (_mm256_set1_epi32(63), _mm256_set1_epi32(0x0f));
    let mut scales = [_mm256_setzero_si256(); 8];
    let mut mins = [_mm256_setzero_si256(); 8];
    for low in 0..4 {
        let scale = rows_i32(packed[low]);
        let min = rows_i32(packed[low + 4]);
        let high = rows_i32(packed[low + 8]);
        let scale_top = _mm256_slli_epi32::<4>(_mm256_srli_epi32::<6>(scale));
        let min_top = _mm256_slli_epi32::<4>(_mm256_srli_epi32::<6>(min));
        scales[low] = _mm256_and_si256(scale, six);
        mins[low] = _mm256_and_si256(min, six);
        scales[low + 4] = _mm256_or_si256(_mm256_and_si256(high, four), scale_top);
        mins[low + 4] = _mm256_or_si256(_mm256_srli_epi32::<4>(high), min_top);
    }
    for scale in &mut scales {
        *scale = _mm256_or_si256(*scale, _mm256_slli_epi32::<16>(*scale));
    }
    Header {
        d: halves(d),
        dmin: halves(dmin),
        scales,
        mins,
    }
}

/// A pair of runs' weights: the low nibbles (the even run), then the high.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn unpack(raw: &[[u8; 32]; 8]) -> [[__m256i; 8]; 2] {
    let mask = _mm256_set1_epi8(0x0f);
    let mut weights = [[_mm256_setzero_si256(); 8]; 2];
    for (index, &lane) in raw.iter().enumerate() {
        let lane: __m256i = bytemuck::cast(lane);
        weights[0][index] = _mm256_and_si256(lane, mask);
        weights[1][index] = _mm256_and_si256(_mm256_srli_epi16::<4>(lane), mask);
    }
    weights
}

/// Adds one 32-element run's dot products times its scale to `sum`.
/// Weight vector `i` holds elements `4i..4i + 4` of every row; the eight
/// `i16` partial sums stay below `8 * 2 * 15 * 128 = 30720` in magnitude.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn run(sum: __m256i, weights: &[__m256i; 8], quants: &[[i8; 4]; 8], scale: __m256i) -> __m256i {
    let mut dots = _mm256_setzero_si256();
    for (&weight, &quad) in weights.iter().zip(quants) {
        let quad = _mm256_set1_epi32(bytemuck::cast(quad));
        dots = _mm256_add_epi16(dots, _mm256_maddubs_epi16(weight, quad));
    }
    _mm256_add_epi32(sum, _mm256_madd_epi16(dots, scale))
}

/// `acc + d8 * (d * sum - dmin * min_sum)` per row, as the portable kernel
/// rounds it: the inner multiply and add fused, as its `mul_add` is.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn combine(acc: __m256, sum: __m256i, header: &Header, block_sums: &[i32; 8], d8: f32) -> __m256 {
    let mut min_sum = _mm256_setzero_si256();
    for (&mins, &block_sum) in header.mins.iter().zip(block_sums) {
        min_sum = _mm256_add_epi32(
            min_sum,
            _mm256_mullo_epi32(mins, _mm256_set1_epi32(block_sum)),
        );
    }
    let min_term = _mm256_mul_ps(header.dmin, _mm256_cvtepi32_ps(min_sum));
    let negated = _mm256_xor_ps(min_term, _mm256_set1_ps(-0.0));
    let term = _mm256_fmadd_ps(header.d, _mm256_cvtepi32_ps(sum), negated);
    _mm256_add_ps(acc, _mm256_mul_ps(term, _mm256_set1_ps(d8)))
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
        let header = header(unit);
        let mut sums = [_mm256_setzero_si256(); N];
        for (pair, raw) in quant_pairs(unit).iter().enumerate() {
            let weights = unpack(raw);
            for (sum, input) in sums.iter_mut().zip(inputs) {
                if let Some(quants) = input.quants.get(block) {
                    let runs: &[[[i8; 4]; 8]; 8] = bytemuck::cast_ref(quants);
                    for (half, weights) in weights.iter().enumerate() {
                        let index = 2 * pair + half;
                        *sum = run(*sum, weights, &runs[index], header.scales[index]);
                    }
                }
            }
        }
        for ((acc, &sum), input) in acc.iter_mut().zip(&sums).zip(inputs) {
            if let (Some(block_sums), Some(scale)) =
                (input.sums.get(block), input.scales.get(block))
            {
                *acc = combine(*acc, sum, &header, block_sums, scale[0]);
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
