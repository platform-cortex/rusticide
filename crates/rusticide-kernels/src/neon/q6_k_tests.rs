// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::BLOCK;
use crate::group_checks;

#[test]
fn prepare_then_read_row_is_identity() {
    group_checks::prepare_then_read_row_is_identity(14, BLOCK);
}

#[test]
fn a_unit_interleaves_the_k_th_block_of_eight_rows() {
    let (unit, block) = group_checks::second_unit_and_block(14, BLOCK);
    assert_eq!(unit[32 * 31 + 12..32 * 31 + 16], block[124..128]);
    assert_eq!(
        unit[1024 + 32 * 2 + 12..1024 + 32 * 2 + 16],
        block[136..140]
    );
    assert_eq!(unit[1536 + 8 * 15 + 3], block[207]);
    assert_eq!(unit[1664 + 6..1664 + 8], block[208..210]);
}

#[test]
fn prepare_leaves_a_matrix_of_partial_groups_or_blocks_untouched() {
    group_checks::prepare_leaves_partial_groups_or_blocks_untouched(14, BLOCK);
}

#[test]
fn read_row_rejects_a_missing_row_or_a_wrong_buffer() {
    group_checks::read_row_rejects_a_missing_row_or_a_wrong_buffer(14, BLOCK);
}
