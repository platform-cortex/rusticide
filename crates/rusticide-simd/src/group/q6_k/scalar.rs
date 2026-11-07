// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The kernel one value at a time: Miri's stand-in for the SIMD bodies,
//! and the reference they are tested against. No sum can overflow: a
//! block's is below `16 * 128 * 16 * 32 * 128 = 2^27` in magnitude.

use super::{D, LANE, QH, QL, SCALES, UNIT};
use crate::group::{Q8Superblocks, ROWS};
use crate::scalar::{f16_at, lossy};

/// Byte `byte` of row `row`'s bits starting at `base`.
const fn interleaved(unit: &[u8; UNIT], base: usize, row: usize, byte: usize) -> u8 {
    unit[base + ROWS * LANE * (byte / LANE) + LANE * row + byte % LANE]
}

/// Weight `element` of row `row`, unpacked as ggml does.
fn weight(unit: &[u8; UNIT], row: usize, element: usize) -> i32 {
    let (half, run, index) = (element / 128, element % 128 / 32, element % 32);
    let low = interleaved(unit, QL, row, 64 * half + 32 * (run % 2) + index);
    let high = interleaved(unit, QH, row, 32 * half + index);
    let nibble = if run < 2 { low & 0x0f } else { low >> 4 };
    i32::from(nibble | (((high >> (2 * run)) & 3) << 4)) - 32
}

fn row_term(unit: &[u8; UNIT], row: usize, quants: &[i8; 256], d8: f32) -> f32 {
    let mut sum = 0_i32;
    for run in 0..16 {
        let dot: i32 = (16 * run..16 * run + 16)
            .map(|element| weight(unit, row, element) * i32::from(quants[element]))
            .sum();
        sum += i32::from(unit[SCALES + ROWS * run + row].cast_signed()) * dot;
    }
    f16_at(unit, D + 2 * row) * d8 * lossy(sum)
}

pub(crate) fn group(units: &[[u8; UNIT]], inputs: &[Q8Superblocks<'_>], out: &mut [[f32; ROWS]]) {
    for (input, rows) in inputs.iter().zip(out) {
        *rows = [-0.0; ROWS];
        for (unit, (quants, scales)) in units.iter().zip(input.quants.iter().zip(input.scales)) {
            for (row, value) in rows.iter_mut().enumerate() {
                *value += row_term(unit, row, quants, scales[0]);
            }
        }
    }
}
