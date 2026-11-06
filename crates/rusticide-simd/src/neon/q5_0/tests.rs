// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{D, QH, UNIT, q5_0_group, scalar};
use crate::group::{Q8Blocks, ROWS};
use crate::test_support::{Blocks, Format, set_f16, token};

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

/// The NEON kernel under the test host's token.
fn neon(group: &[[u8; UNIT]], inputs: &[Q8Blocks<'_>], out: &mut [[f32; ROWS]]) {
    q5_0_group(token(), group, inputs, out);
}

const Q5_0: Format<UNIT, Blocks> = Format {
    kernels: [neon, scalar::group],
    random_units,
    set_specials,
};

#[test]
fn the_neon_body_matches_the_scalar_body() {
    Q5_0.the_simd_body_matches_the_scalar_body(0x0005_00a5);
}

#[test]
fn every_token_is_computed_as_if_alone() {
    Q5_0.every_token_is_computed_as_if_alone(53);
}

#[test]
fn the_shorter_of_group_and_input_bounds_the_blocks() {
    Q5_0.the_shorter_of_group_and_input_bounds_the_blocks(59);
}

#[test]
fn special_scales_match_the_scalar_body() {
    Q5_0.special_scales_match_the_scalar_body(61);
}

#[test]
fn a_group_at_an_odd_address_gives_the_same_products() {
    Q5_0.a_group_at_an_odd_address_gives_the_same_products(67);
}

#[test]
fn the_largest_sums_are_exact() {
    let mut unit = [0; UNIT];
    for row in 0..ROWS {
        set_f16(&mut unit, D + 2 * row, 0x3c00);
    }
    let lowest = [[-127; 32]];
    let view = [Q8Blocks::new(&[1.0], &lowest).unwrap()];
    let [native, reference] = Q5_0.both(&[unit], &view);
    assert_eq!(native, [[65_024.0_f32.to_bits(); ROWS]]);
    assert_eq!(native, reference);
    unit[QH..].fill(0xff);
    let [native, reference] = Q5_0.both(&[unit], &view);
    assert_eq!(native, [[(-60_960.0_f32).to_bits(); ROWS]]);
    assert_eq!(native, reference);
}
