// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The kernel on AVX2: a unit's weight bytes are already the eight signed
//! weight vectors.

use super::UNIT;
use crate::avx2::blocks::{self, Unpacked};
use crate::group::q8_0::{D, QS};
use crate::group::{Q8Blocks, ROWS};

// `unpack` hard-codes these offsets in its pattern.
const _: () = assert!(D == 0 && QS == 16 && UNIT == QS + 8 * 32);

fn unpack(unit: &[u8; UNIT]) -> Unpacked {
    let lanes: &[[u8; 16]; 17] = bytemuck::cast_ref(unit);
    let [d, weights @ ..] = lanes;
    let weights: &[[u8; 32]; 8] = bytemuck::cast_ref(weights);
    (*d, weights.map(bytemuck::cast))
}

#[target_feature(enable = "avx2,fma")]
pub(super) fn group(units: &[[u8; UNIT]], inputs: &[Q8Blocks<'_>], out: &mut [[f32; ROWS]]) {
    blocks::group(units, inputs, out, unpack);
}
