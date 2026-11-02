// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{D, QH, QL, SCALES, UNIT, q6_k_group, scalar};
use crate::neon::group::ROWS;
use crate::neon::test_support::{Activation, Format, set_f16};

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

const Q6_K: Format<UNIT> = Format {
    kernels: [q6_k_group, scalar::group],
    random_units,
    set_specials,
};

#[test]
fn the_neon_body_matches_the_scalar_body() {
    Q6_K.the_neon_body_matches_the_scalar_body(0x0006_00a6);
}

#[test]
fn every_token_is_computed_as_if_alone() {
    Q6_K.every_token_is_computed_as_if_alone(17);
}

#[test]
fn the_shorter_of_group_and_input_bounds_the_blocks() {
    Q6_K.the_shorter_of_group_and_input_bounds_the_blocks(19);
}

#[test]
fn special_scales_match_the_scalar_body() {
    Q6_K.special_scales_match_the_scalar_body(23);
}

#[test]
fn a_group_at_an_odd_address_gives_the_same_products() {
    Q6_K.a_group_at_an_odd_address_gives_the_same_products(29);
}

#[test]
fn the_largest_sums_are_exact() {
    let mut unit = [0; UNIT];
    unit[SCALES..D].fill(0x80);
    for row in 0..ROWS {
        set_f16(&mut unit, D + 2 * row, 0x3c00);
    }
    let lowest = Activation::from_quants(vec![[i8::MIN; 256]], vec![1.0]);
    let [native, reference] = Q6_K.both(&[unit], &[lowest.view()]);
    let expected = -134_217_728.0_f32;
    assert_eq!(native, [[expected.to_bits(); ROWS]]);
    assert_eq!(native, reference);
    unit[QL..QH].fill(0xff);
    unit[QH..SCALES].fill(0xff);
    let [native, reference] = Q6_K.both(&[unit], &[lowest.view()]);
    assert_eq!(native, [[(31.0_f32 / 32.0 * -expected).to_bits(); ROWS]]);
    assert_eq!(native, reference);
}
