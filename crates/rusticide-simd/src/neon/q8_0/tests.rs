// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{D, QS, UNIT, q8_0_group, scalar};
use crate::neon::group::{Q8Blocks, ROWS};
use crate::neon::test_support::{Blocks, Format, set_f16};

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

const Q8_0: Format<UNIT, Blocks> = Format {
    kernels: [q8_0_group, scalar::group],
    random_units,
    set_specials,
};

#[test]
fn the_neon_body_matches_the_scalar_body() {
    Q8_0.the_neon_body_matches_the_scalar_body(0x0008_00a8);
}

#[test]
fn every_token_is_computed_as_if_alone() {
    Q8_0.every_token_is_computed_as_if_alone(37);
}

#[test]
fn the_shorter_of_group_and_input_bounds_the_blocks() {
    Q8_0.the_shorter_of_group_and_input_bounds_the_blocks(41);
}

#[test]
fn special_scales_match_the_scalar_body() {
    Q8_0.special_scales_match_the_scalar_body(43);
}

#[test]
fn a_group_at_an_odd_address_gives_the_same_products() {
    Q8_0.a_group_at_an_odd_address_gives_the_same_products(47);
}

#[test]
fn the_largest_sums_are_exact() {
    let mut unit = [0x80; UNIT];
    for row in 0..ROWS {
        set_f16(&mut unit, D + 2 * row, 0x3c00);
    }
    let lowest = [[i8::MIN; 32]];
    let view = [Q8Blocks::new(&[1.0], &lowest)];
    let [native, reference] = Q8_0.both(&[unit], &view);
    assert_eq!(native, [[524_288.0_f32.to_bits(); ROWS]]);
    assert_eq!(native, reference);
    unit[QS..].fill(0x7f);
    let [native, reference] = Q8_0.both(&[unit], &view);
    assert_eq!(native, [[(-520_192.0_f32).to_bits(); ROWS]]);
    assert_eq!(native, reference);
}
