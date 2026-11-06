// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::BLOCK;
use crate::group_checks;

#[test]
fn prepare_then_read_row_is_identity() {
    group_checks::prepare_then_read_row_is_identity(12, BLOCK);
}

#[test]
fn a_unit_interleaves_the_k_th_block_of_eight_rows() {
    let (unit, block) = group_checks::second_unit_and_block(12, BLOCK);
    assert_eq!(unit[6..8], block[0..2]);
    assert_eq!(unit[22..24], block[2..4]);
    assert_eq!(unit[32 + 8 * 11 + 3], block[15]);
    assert_eq!(
        unit[128 + 32 * 31 + 12..128 + 32 * 31 + 16],
        block[140..144]
    );
}

#[test]
fn prepare_leaves_a_matrix_of_partial_groups_or_blocks_untouched() {
    group_checks::prepare_leaves_partial_groups_or_blocks_untouched(12, BLOCK);
}

#[test]
fn read_row_rejects_a_missing_row_or_a_wrong_buffer() {
    group_checks::read_row_rejects_a_missing_row_or_a_wrong_buffer(12, BLOCK);
}
