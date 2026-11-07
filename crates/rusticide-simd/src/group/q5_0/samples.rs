// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `Q5_0` units every SIMD body is tested on.

use super::{D, QH, UNIT, scalar};
use crate::group::ROWS;
use crate::test_support::{Blocks, Format, Input, Kernel, random_d_units, set_d_specials};

/// The checks for the SIMD body `native`, against the scalar body.
pub(crate) const fn format(native: Kernel<UNIT, Blocks>) -> Format<UNIT, Blocks> {
    Format {
        kernels: [native, scalar::group],
        random_units: random_d_units::<UNIT, D>,
        set_specials: set_d_specials::<UNIT, D>,
    }
}

/// Weights all `-16` (no fifth bit), then all `15`, against the quant
/// `-127`: the largest sums of either sign, exactly.
pub(crate) fn the_largest_sums_are_exact(format: &Format<UNIT, Blocks>) {
    let mut unit = [0; UNIT];
    set_d_specials::<UNIT, D>(&mut unit, &[0x3c00; ROWS]);
    let lowest = Blocks {
        scales: vec![1.0],
        quants: vec![[-127; 32]],
    };
    let view = [lowest.view()];
    let [native, reference] = format.both(&[unit], &view);
    assert_eq!(native, [[65_024.0_f32.to_bits(); ROWS]]);
    assert_eq!(native, reference);
    unit[QH..].fill(0xff);
    let [native, reference] = format.both(&[unit], &view);
    assert_eq!(native, [[(-60_960.0_f32).to_bits(); ROWS]]);
    assert_eq!(native, reference);
}
