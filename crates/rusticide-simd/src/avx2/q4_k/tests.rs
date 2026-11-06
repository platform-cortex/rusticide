// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{UNIT, q4_k_group};
use crate::group::q4_k::samples;
use crate::group::{Q8Superblocks, ROWS};
use crate::test_support::{Format, token};

/// The AVX2 kernel under the test host's token.
fn avx2(group: &[[u8; UNIT]], inputs: &[Q8Superblocks<'_>], out: &mut [[f32; ROWS]]) {
    q4_k_group(token(), group, inputs, out);
}

const Q4_K: Format<UNIT> = samples::format(avx2);

#[test]
fn the_avx2_body_matches_the_scalar_body() {
    Q4_K.the_simd_body_matches_the_scalar_body(0x0004_00a4);
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

#[test]
fn extreme_bytes_match_the_scalar_body() {
    Q4_K.extreme_bytes_match_the_scalar_body();
}

#[test]
fn wrapping_sums_match_the_scalar_body() {
    samples::wrapping_sums_match_the_scalar_body(&Q4_K, 17);
}
