// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The kernel one value at a time: Miri's stand-in for the NEON body, and
//! the reference the NEON body is tested against.

use super::{D, LANE, QS, UNIT};
use crate::group::{Q8Blocks, ROWS};
use crate::neon::scalar::blocks;

/// Weight `element` of row `row`.
fn weight(unit: &[u8; UNIT], row: usize, element: usize) -> i32 {
    let byte = unit[QS + ROWS * LANE * (element / LANE) + LANE * row + element % LANE];
    i32::from(byte.cast_signed())
}

pub(super) fn group(units: &[[u8; UNIT]], inputs: &[Q8Blocks<'_>], out: &mut [[f32; ROWS]]) {
    blocks(units, inputs, out, D, weight);
}
