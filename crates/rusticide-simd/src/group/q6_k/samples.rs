// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `Q6_K` units every SIMD body is tested on.

use super::{D, QH, QL, SCALES, UNIT, scalar};
use crate::group::ROWS;
use crate::test_support::{Activation, Format, Input, Kernel, set_f16};

/// Random units whose `d` are finite f16 values in `[0, 2)`.
fn random_units(rng: &mut fastrand::Rng, blocks: usize) -> Vec<[u8; UNIT]> {
    (0..blocks)
        .map(|_| {
            let mut unit: [u8; UNIT] = core::array::from_fn(|_| rng.u8(..));
            for row in 0..ROWS {
                set_f16(&mut unit, D + 2 * row, rng.u16(..0x4000));
            }
            unit
        })
        .collect()
}

/// Each row's `d` gets its special.
fn set_specials(unit: &mut [u8; UNIT], specials: &[u16; ROWS]) {
    for (row, &bits) in specials.iter().enumerate() {
        set_f16(unit, D + 2 * row, bits);
    }
}

/// The checks for the SIMD body `native`, against the scalar body.
pub(crate) const fn format(native: Kernel<UNIT, Activation>) -> Format<UNIT> {
    Format {
        kernels: [native, scalar::group],
        random_units,
        set_specials,
    }
}

/// The scale `-128` on every run, against the quant `-127`: weights all
/// `-32`, then all `31`, give the largest sums of either sign exactly.
pub(crate) fn the_largest_sums_are_exact(format: &Format<UNIT>) {
    let mut unit = [0; UNIT];
    unit[SCALES..D].fill(0x80);
    for row in 0..ROWS {
        set_f16(&mut unit, D + 2 * row, 0x3c00);
    }
    let lowest = Activation::from_quants(vec![[-127; 256]], vec![1.0]);
    let [native, reference] = format.both(&[unit], &[lowest.view()]);
    let expected = -133_169_152.0_f32;
    assert_eq!(native, [[expected.to_bits(); ROWS]]);
    assert_eq!(native, reference);
    unit[QL..QH].fill(0xff);
    unit[QH..SCALES].fill(0xff);
    let [native, reference] = format.both(&[unit], &[lowest.view()]);
    assert_eq!(native, [[(31.0_f32 / 32.0 * -expected).to_bits(); ROWS]]);
    assert_eq!(native, reference);
}
