// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::dot_group_with;
use crate::group_checks;
use crate::portable;
use crate::q8_input::Q8Input;

/// The portable `Q8_0` kernel has no batched form: one dot per input.
fn portable_rows(row: &[u8], inputs: &[Q8Input], out: &mut [f32]) {
    portable::Q8_0.dot_rows(row, inputs, out);
}

#[test]
fn neon_q8_0_matches_portable() {
    group_checks::simd_matches_portable(8, portable_rows);
}

#[test]
fn a_batch_is_bit_identical_to_its_tokens_one_at_a_time() {
    group_checks::a_batch_is_bit_identical_to_its_tokens_one_at_a_time(8);
}

#[test]
fn no_inputs_leave_the_output_untouched() {
    group_checks::no_inputs_leave_the_output_untouched(8);
}

#[test]
fn the_adapter_runs_q8_0_on_neon_in_groups_of_eight() {
    group_checks::runs_in_groups_of_eight(8);
}

#[test]
fn a_minus_128_activation_matches_portable() {
    group_checks::a_minus_128_activation_matches_portable(8);
}

#[test]
fn wrapper_without_token_matches_portable() {
    group_checks::wrapper_without_token_matches_portable(8, |group, inputs, views, out| {
        dot_group_with(None, group, inputs, views, out);
    });
}
