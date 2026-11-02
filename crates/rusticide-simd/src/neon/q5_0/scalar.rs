// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The kernel one value at a time: Miri's stand-in for the NEON body, and
//! the reference the NEON body is tested against.

use super::{D, LANE, QH, QS, UNIT};
use crate::group::{Q8Blocks, ROWS};
use crate::neon::scalar::blocks;

/// Weight `element` of row `row`, unpacked as ggml does.
fn weight(unit: &[u8; UNIT], row: usize, element: usize) -> i32 {
    let byte = element % 16;
    let packed = unit[QS + ROWS * LANE * (byte / LANE) + LANE * row + byte % LANE];
    let nibble = if element < 16 {
        packed & 0x0f
    } else {
        packed >> 4
    };
    let high = u32::from_le_bytes(core::array::from_fn(|i| unit[QH + LANE * row + i]));
    i32::from(nibble) + 16 * i32::from((high >> element) & 1 == 1) - 16
}

pub(super) fn group(units: &[[u8; UNIT]], inputs: &[Q8Blocks<'_>], out: &mut [[f32; ROWS]]) {
    blocks(units, inputs, out, D, weight);
}
