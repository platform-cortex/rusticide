// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The kernel on NEON: a unit's weight bytes are already the sixteen
//! weight vectors.

use super::{D, QS, UNIT};
use crate::group::{Q8Blocks, ROWS};
use crate::neon::blocks::{self, Unpacked};

// `unpack` hard-codes these offsets in its pattern.
const _: () = assert!(D == 0 && QS == 16 && UNIT == QS + 16 * 16);

fn unpack(unit: &[u8; UNIT]) -> Unpacked {
    let lanes: &[[u8; 16]; 17] = bytemuck::cast_ref(unit);
    let [d, weights @ ..] = lanes;
    (*d, bytemuck::cast(*weights))
}

#[target_feature(enable = "neon,dotprod")]
pub(super) fn group(units: &[[u8; UNIT]], inputs: &[Q8Blocks<'_>], out: &mut [[f32; ROWS]]) {
    blocks::group(units, inputs, out, unpack);
}
