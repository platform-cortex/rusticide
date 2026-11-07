// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The kernel on NEON. A weight is its nibble with `0xf0` set unless its
//! fifth bit is: the nibble plus 16 times the bit, minus 16, as an i8. Each
//! lane's fifth bits come from its row's byte by a table lookup and a bit
//! test.

use core::arch::aarch64::{
    int8x16_t, uint8x16_t, vandq_u8, vbicq_u8, vdupq_n_s8, vdupq_n_u8, vorrq_u8, vqtbl1q_u8,
    vreinterpretq_s8_u8, vshrq_n_u8, vtstq_u8,
};

use super::UNIT;
use crate::group::q5_0::{D, QH, QS};
use crate::group::{Q8Blocks, ROWS};
use crate::neon::blocks::{self, Unpacked};

// `unpack` hard-codes these offsets in its pattern.
const _: () = assert!(D == 0 && QH == 16 && QS == 48 && UNIT == QS + 8 * 16);

/// For fifth-bit byte `j`: each four-byte lane repeats byte `j` of its
/// row's word.
const BYTE: [[u8; 16]; 4] = [
    [0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12],
    [1, 1, 1, 1, 5, 5, 5, 5, 9, 9, 9, 9, 13, 13, 13, 13],
    [2, 2, 2, 2, 6, 6, 6, 6, 10, 10, 10, 10, 14, 14, 14, 14],
    [3, 3, 3, 3, 7, 7, 7, 7, 11, 11, 11, 11, 15, 15, 15, 15],
];

/// The bit of a lane's weight in its fifth-bit byte, for an even and an
/// odd chunk of four weights.
const BIT: [[u8; 16]; 2] = [
    [1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8],
    [
        16, 32, 64, 128, 16, 32, 64, 128, 16, 32, 64, 128, 16, 32, 64, 128,
    ],
];

/// The weights of chunk `chunk` (elements `4 * chunk..`) of four rows,
/// from their nibbles and the rows' fifth-bit words.
#[inline]
#[target_feature(enable = "neon,dotprod")]
fn with_fifth(nibbles: uint8x16_t, words: uint8x16_t, chunk: usize) -> int8x16_t {
    let bytes = vqtbl1q_u8(words, bytemuck::cast(BYTE[chunk / 2]));
    let set = vtstq_u8(bytes, bytemuck::cast(BIT[chunk % 2]));
    vreinterpretq_s8_u8(vorrq_u8(nibbles, vbicq_u8(vdupq_n_u8(0xf0), set)))
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
fn unpack(unit: &[u8; UNIT]) -> Unpacked {
    let lanes: &[[u8; 16]; 11] = bytemuck::cast_ref(unit);
    let [d, low_words, high_words, packed @ ..] = lanes;
    let words: [uint8x16_t; 2] = [bytemuck::cast(*low_words), bytemuck::cast(*high_words)];
    let mut weights = [vdupq_n_s8(0); 16];
    let (low, high) = weights.split_at_mut(8);
    for (lane, ((&bytes, low), high)) in packed.iter().zip(low).zip(high).enumerate() {
        let (chunk, quad) = (lane / 2, lane % 2);
        let bytes: uint8x16_t = bytemuck::cast(bytes);
        *low = with_fifth(vandq_u8(bytes, vdupq_n_u8(0x0f)), words[quad], chunk);
        *high = with_fifth(vshrq_n_u8::<4>(bytes), words[quad], chunk + 4);
    }
    (*d, weights)
}

#[target_feature(enable = "neon,dotprod")]
pub(super) fn group(units: &[[u8; UNIT]], inputs: &[Q8Blocks<'_>], out: &mut [[f32; ROWS]]) {
    // A `#[target_feature]` fn does not implement `Fn`, so `unpack` cannot
    // be passed by name; the closure, built where the features are enabled,
    // can.
    blocks::group(units, inputs, out, |unit| unpack(unit));
}
