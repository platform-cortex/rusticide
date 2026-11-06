// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The scalar body every 32-element block format shares: Miri's stand-in
//! for the NEON bodies, and the reference they are tested against.

use crate::group::{Q8Blocks, ROWS};
use crate::scalar::{f16_at, lossy};

fn block_term<const UNIT: usize>(
    unit: &[u8; UNIT],
    row: usize,
    quants: &[i8; 32],
    d8: f32,
    d: usize,
    weight: fn(&[u8; UNIT], usize, usize) -> i32,
) -> f32 {
    let sum: i32 = (0..32)
        .map(|element| weight(unit, row, element) * i32::from(quants[element]))
        .sum();
    f16_at(unit, d + 2 * row) * d8 * lossy(sum)
}

/// A 32-element block format's kernel: `weight(unit, row, element)` is a
/// row's signed weight and its `d` sits at `d + 2 * row`. No sum can
/// overflow: a block's is at most `32 * 128 * 128 = 2^19` in magnitude.
pub(crate) fn blocks<const UNIT: usize>(
    units: &[[u8; UNIT]],
    inputs: &[Q8Blocks<'_>],
    out: &mut [[f32; ROWS]],
    d: usize,
    weight: fn(&[u8; UNIT], usize, usize) -> i32,
) {
    for (input, rows) in inputs.iter().zip(out) {
        *rows = [-0.0; ROWS];
        for (unit, (quants, &d8)) in units.iter().zip(input.quants.iter().zip(input.scales)) {
            for (row, value) in rows.iter_mut().enumerate() {
                *value += block_term(unit, row, quants, d8, d, weight);
            }
        }
    }
}
