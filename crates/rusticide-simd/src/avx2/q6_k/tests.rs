// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{UNIT, q6_k_group};
use crate::group::q6_k::samples;
use crate::group::{Q8Superblocks, ROWS};
use crate::test_support::{Format, token};

/// The AVX2 kernel under the test host's token.
fn avx2(group: &[[u8; UNIT]], inputs: &[Q8Superblocks<'_>], out: &mut [[f32; ROWS]]) {
    q6_k_group(token(), group, inputs, out);
}

const Q6_K: Format<UNIT> = samples::format(avx2);

#[test]
fn the_avx2_body_matches_the_scalar_body() {
    Q6_K.the_simd_body_matches_the_scalar_body(0x0006_00a6);
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
fn extreme_bytes_match_the_scalar_body() {
    Q6_K.extreme_bytes_match_the_scalar_body();
}

#[test]
fn the_largest_sums_are_exact() {
    samples::the_largest_sums_are_exact(&Q6_K);
}
