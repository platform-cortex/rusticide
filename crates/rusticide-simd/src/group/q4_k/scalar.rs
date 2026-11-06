// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The kernel one value at a time: Miri's stand-in for the SIMD bodies,
//! and the reference they are tested against. Integer arithmetic wraps as
//! theirs does, so no input can make them differ or this one panic.

use super::{D, DMIN, LANE, QS, SCALES, UNIT};
use crate::group::{Q8Superblocks, ROWS};
use crate::scalar::{f16_at, lossy};

/// Row `row`'s eight 6-bit scales and mins, unpacked as ggml does.
fn scales_and_mins(unit: &[u8; UNIT], row: usize) -> ([i32; 8], [i32; 8]) {
    let byte = |index: usize| unit[SCALES + ROWS * index + row];
    let mut scales = [0; 8];
    let mut mins = [0; 8];
    for low in 0..4 {
        let (scale, min, high) = (byte(low), byte(low + 4), byte(low + 8));
        scales[low] = i32::from(scale & 63);
        mins[low] = i32::from(min & 63);
        scales[low + 4] = i32::from((high & 0x0f) | ((scale >> 6) << 4));
        mins[low + 4] = i32::from((high >> 4) | ((min >> 6) << 4));
    }
    (scales, mins)
}

/// Quant `element` of 32-element run `run` of row `row`.
fn quant(unit: &[u8; UNIT], row: usize, run: usize, element: usize) -> i32 {
    let byte = 32 * (run / 2) + element;
    let packed = unit[QS + ROWS * LANE * (byte / LANE) + LANE * row + byte % LANE];
    i32::from(if run.is_multiple_of(2) {
        packed & 0x0f
    } else {
        packed >> 4
    })
}

fn row_term(unit: &[u8; UNIT], row: usize, quants: &[i8; 256], sums: &[i32; 8], d8: f32) -> f32 {
    let (scales, mins) = scales_and_mins(unit, row);
    let mut sum = 0_i32;
    let mut min_sum = 0_i32;
    for run in 0..8 {
        let mut dot = 0_i32;
        for element in 0..32 {
            let q = i32::from(quants[32 * run + element]);
            dot = dot.wrapping_add(quant(unit, row, run, element) * q);
        }
        sum = sum.wrapping_add(scales[run].wrapping_mul(dot));
        min_sum = min_sum.wrapping_add(mins[run].wrapping_mul(sums[run]));
    }
    let d = f16_at(unit, D + 2 * row);
    let dmin = f16_at(unit, DMIN + 2 * row);
    d8 * d.mul_add(lossy(sum), -(dmin * lossy(min_sum)))
}

pub(crate) fn group(units: &[[u8; UNIT]], inputs: &[Q8Superblocks<'_>], out: &mut [[f32; ROWS]]) {
    for (input, rows) in inputs.iter().zip(out) {
        *rows = [-0.0; ROWS];
        let blocks = input.quants.iter().zip(input.sums).zip(input.scales);
        for (unit, ((quants, sums), scales)) in units.iter().zip(blocks) {
            for (row, value) in rows.iter_mut().enumerate() {
                *value += row_term(unit, row, quants, sums, scales[0]);
            }
        }
    }
}
