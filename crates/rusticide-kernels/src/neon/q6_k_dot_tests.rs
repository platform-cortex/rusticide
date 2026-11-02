// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::neon::group_checks;
use crate::portable::int_dot;

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
