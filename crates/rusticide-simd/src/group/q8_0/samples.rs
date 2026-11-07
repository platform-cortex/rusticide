// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `Q8_0` units every SIMD body is tested on.

use super::{D, QS, UNIT, scalar};
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

/// Weights all `-128`, then all `127`, against the quant `-127`: the
/// largest sums of either sign, exactly.
pub(crate) fn the_largest_sums_are_exact(format: &Format<UNIT, Blocks>) {
    let mut unit = [0x80; UNIT];
    set_d_specials::<UNIT, D>(&mut unit, &[0x3c00; ROWS]);
    let lowest = Blocks {
        scales: vec![1.0],
        quants: vec![[-127; 32]],
    };
    let view = [lowest.view()];
    let [native, reference] = format.both(&[unit], &view);
    assert_eq!(native, [[520_192.0_f32.to_bits(); ROWS]]);
    assert_eq!(native, reference);
    unit[QS..].fill(0x7f);
    let [native, reference] = format.both(&[unit], &view);
    assert_eq!(native, [[(-516_128.0_f32).to_bits(); ROWS]]);
    assert_eq!(native, reference);
}

/// Weights all `-128`, whose magnitude `128` only fits `vpmaddubsw`'s
/// unsigned operand, against the quants `127`, `-127` and the two
/// alternating: each pair sums to `2 * 128 * 127 = 32512` at most, within
/// `i16`.
pub(crate) fn minus_128_weights_against_127_are_exact(format: &Format<UNIT, Blocks>) {
    let mut unit = [0x80; UNIT];
    set_d_specials::<UNIT, D>(&mut unit, &[0x3c00; ROWS]);
    let alternating = core::array::from_fn(|element| if element % 2 == 0 { 127 } else { -127 });
    let activations = [[127; 32], [-127; 32], alternating].map(|quants| Blocks {
        scales: vec![1.0],
        quants: vec![quants],
    });
    let views: Vec<_> = activations.iter().map(Input::view).collect();
    let [native, reference] = format.both(&[unit], &views);
    let expected = [-520_192.0_f32, 520_192.0, 0.0].map(|value| [value.to_bits(); ROWS]);
    assert_eq!(native, expected);
    assert_eq!(native, reference);
}
