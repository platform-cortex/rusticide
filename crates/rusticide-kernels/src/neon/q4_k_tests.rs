// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{BLOCK, prepare, read_row};

const ROWS: usize = 16;

fn random_rows(blocks: usize, seed: u64) -> (Vec<u8>, usize) {
    let mut rng = fastrand::Rng::with_seed(seed);
    let row_bytes = blocks * BLOCK;
    (
        (0..ROWS * row_bytes).map(|_| rng.u8(..)).collect(),
        row_bytes,
    )
}

#[test]
fn prepare_then_read_row_is_identity() {
    for blocks in [1, 3] {
        let (original, row_bytes) = random_rows(blocks, 5);
        let mut data = original.clone();
        assert!(prepare(&mut data, row_bytes));
        assert_ne!(data, original);
        for (index, expected) in original.chunks_exact(row_bytes).enumerate() {
            let mut row = vec![0; row_bytes];
            assert!(read_row(&data, row_bytes, index, &mut row));
            assert_eq!(row, expected, "row {index} of {blocks} blocks");
        }
    }
}

#[test]
fn a_unit_interleaves_the_k_th_block_of_eight_rows() {
    let (original, row_bytes) = random_rows(2, 9);
    let mut data = original.clone();
    assert!(prepare(&mut data, row_bytes));
    let second_unit = &data[1152..2304];
    let row_three_block_one = &original[3 * row_bytes + BLOCK..3 * row_bytes + 2 * BLOCK];
    assert_eq!(second_unit[6..8], row_three_block_one[0..2]);
    assert_eq!(second_unit[22..24], row_three_block_one[2..4]);
    assert_eq!(second_unit[32 + 8 * 11 + 3], row_three_block_one[15]);
    assert_eq!(
        second_unit[128 + 32 * 31 + 12..128 + 32 * 31 + 16],
        row_three_block_one[140..144]
    );
}

#[test]
fn prepare_leaves_a_matrix_of_partial_groups_or_blocks_untouched() {
    let (original, row_bytes) = random_rows(1, 3);
    for (len, width) in [
        (7 * row_bytes, row_bytes),
        (original.len(), 100),
        (original.len(), 0),
    ] {
        let mut data = original[..len].to_vec();
        assert!(
            !prepare(&mut data, width),
            "{len} bytes of {width}-byte rows"
        );
        assert_eq!(data, original[..len]);
    }
    let overflowing = BLOCK * (usize::MAX / BLOCK / 4);
    assert!(!prepare(&mut [0; 8], overflowing));
    assert!(prepare(&mut [], row_bytes));
}

#[test]
fn read_row_rejects_a_missing_row_or_a_wrong_buffer() {
    let (mut data, row_bytes) = random_rows(1, 4);
    assert!(prepare(&mut data, row_bytes));
    let mut row = vec![7; row_bytes];
    assert!(!read_row(&data, row_bytes, ROWS, &mut row));
    assert!(!read_row(&data, row_bytes, usize::MAX, &mut row));
    assert!(!read_row(&data, row_bytes, 0, &mut row[1..]));
    assert!(!read_row(&data, 0, 0, &mut []));
    assert!(!read_row(&data, 100, 0, &mut row[..100]));
    assert_eq!(row, vec![7; row_bytes]);
}
