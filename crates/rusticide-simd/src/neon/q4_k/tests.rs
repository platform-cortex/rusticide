// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{D, DMIN, UNIT, q4_k_group, scalar};
use crate::neon::group::ROWS;
use crate::neon::test_support::{Format, set_f16};

/// Random units whose `d` and `dmin` are finite f16 values in `[0, 2)`.
fn random_units(rng: &mut fastrand::Rng, blocks: usize) -> Vec<[u8; UNIT]> {
    (0..blocks)
        .map(|_| {
            let mut unit: [u8; UNIT] = core::array::from_fn(|_| rng.u8(..));
            for row in 0..ROWS {
                set_f16(&mut unit, D + 2 * row, rng.u16(..0x4000));
                set_f16(&mut unit, DMIN + 2 * row, rng.u16(..0x4000));
            }
            unit
        })
        .collect()
}

/// Each row's `d` gets its special and its `dmin` that of the row 3 on.
fn set_specials(unit: &mut [u8; UNIT], specials: &[u16; ROWS]) {
    for (row, &bits) in specials.iter().enumerate() {
        set_f16(unit, D + 2 * row, bits);
        set_f16(unit, DMIN + 2 * row, specials[(row + 3) % ROWS]);
    }
}

const Q4_K: Format<UNIT> = Format {
    kernels: [q4_k_group, scalar::group],
    random_units,
    set_specials,
};

#[test]
fn the_neon_body_matches_the_scalar_body() {
    Q4_K.the_neon_body_matches_the_scalar_body(0x0004_00a4);
}

#[test]
fn every_token_is_computed_as_if_alone() {
    Q4_K.every_token_is_computed_as_if_alone(7);
}

#[test]
fn the_shorter_of_group_and_input_bounds_the_blocks() {
    Q4_K.the_shorter_of_group_and_input_bounds_the_blocks(11);
}

#[test]
fn special_scales_match_the_scalar_body() {
    Q4_K.special_scales_match_the_scalar_body(13);
}

#[test]
fn a_group_at_an_odd_address_gives_the_same_products() {
    Q4_K.a_group_at_an_odd_address_gives_the_same_products(31);
}
