// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_simd::Dotprod;
use rusticide_simd::neon::q6_k::q6_k_group;

use super::KERNEL;
use crate::group_views::GroupViews;
use crate::neon::{group_checks, group_dot};
use crate::portable::int_dot;
use crate::q8_input::Q8Input;

#[test]
fn neon_q6_k_matches_portable() {
    group_checks::neon_matches_portable(14, int_dot::q6_k_rows);
}

#[test]
fn a_batch_is_bit_identical_to_its_tokens_one_at_a_time() {
    group_checks::a_batch_is_bit_identical_to_its_tokens_one_at_a_time(14);
}

#[test]
fn no_inputs_leave_the_output_untouched() {
    group_checks::no_inputs_leave_the_output_untouched(14);
}

#[test]
fn the_adapter_runs_q6_k_on_neon_in_groups_of_eight() {
    group_checks::runs_in_groups_of_eight(14);
}

#[test]
fn a_minus_128_activation_matches_portable() {
    group_checks::a_minus_128_activation_matches_portable(14);
}

fn untokened(group: &[u8], inputs: &[Q8Input], views: &GroupViews<'_>, out: &mut [f32]) {
    let views = views.superblocks();
    group_dot::dot_group(
        &KERNEL,
        None::<Dotprod>,
        q6_k_group,
        views,
        group,
        inputs,
        out,
    );
}

#[test]
fn without_a_token_matches_portable() {
    group_checks::without_a_token_matches_portable(14, untokened);
}
