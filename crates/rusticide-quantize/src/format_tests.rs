// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn table_knows_the_existing_formats() {
    assert_eq!(block_format(2).map(|f| f.name), Some("Q4_0"));
    assert_eq!(
        block_format(12).map(|f| (f.block_size, f.block_bytes)),
        Some((256, 144))
    );
    assert_eq!(block_format(14).map(|f| f.name), Some("Q6_K"));
}

#[test]
fn table_knows_q5_0_and_q8_0() {
    assert_eq!(
        block_format(6).map(|f| (f.name, f.block_size, f.block_bytes)),
        Some(("Q5_0", 32, 22))
    );
    assert_eq!(
        block_format(8).map(|f| (f.name, f.block_size, f.block_bytes)),
        Some(("Q8_0", 32, 34))
    );
}

#[test]
fn formats_lists_every_row_with_a_distinct_type_id() {
    let mut ids: Vec<u32> = formats().iter().map(|f| f.type_id).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids, [2, 6, 8, 12, 14]);
}

#[test]
fn getters_expose_the_row() {
    let format = block_format(12).unwrap();
    assert_eq!(
        (
            format.name(),
            format.type_id(),
            format.block_size(),
            format.block_bytes()
        ),
        ("Q4_K", 12, 256, 144)
    );
}

#[test]
fn unknown_type_id_has_no_format() {
    assert!(block_format(99).is_none());
}

#[test]
fn byte_len_counts_whole_blocks() {
    assert_eq!(block_format(2).unwrap().byte_len(64), Some(36));
    assert_eq!(block_format(2).unwrap().byte_len(33), Some(36));
}

// No table row has more bytes than elements per block, so overflow needs a
// synthetic format.
const WIDER_THAN_ITS_BLOCK: BlockFormat = BlockFormat {
    name: "WIDE",
    type_id: 0,
    block_size: 1,
    block_bytes: 2,
    dequantize_block: |_, _| {},
};

#[test]
fn byte_len_overflow_is_none() {
    assert_eq!(WIDER_THAN_ITS_BLOCK.byte_len(usize::MAX), None);
}

#[test]
fn dequantize_rejects_a_partial_block() {
    let err = block_format(12)
        .unwrap()
        .dequantize(&[0; 144], &[255])
        .unwrap_err();
    assert!(matches!(
        err,
        QuantizeError::UnalignedShape {
            format: "Q4_K",
            numel: 255,
            block: 256
        }
    ));
}

#[test]
fn dequantize_rejects_short_data() {
    let err = block_format(14)
        .unwrap()
        .dequantize(&[0; 10], &[256])
        .unwrap_err();
    assert!(matches!(
        err,
        QuantizeError::DataTooShort {
            format: "Q6_K",
            got: 10,
            needed: 210
        }
    ));
}

#[cfg(target_pointer_width = "64")]
#[test]
fn dequantize_of_an_overflowing_byte_length_is_too_short() {
    let err = WIDER_THAN_ITS_BLOCK
        .dequantize(&[], &[1 << 63])
        .unwrap_err();
    assert!(matches!(
        err,
        QuantizeError::DataTooShort {
            format: "WIDE",
            got: 0,
            needed: usize::MAX
        }
    ));
}

/// Deterministic bytes with bit 6 clear, so every embedded f16 scale is finite.
fn pseudo_random_bytes(len: usize, seed: u32) -> Vec<u8> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8 & 0xBF
        })
        .collect()
}

fn reference_dot(values: &[f32], input: &[f32]) -> f32 {
    values.iter().zip(input).map(|(a, b)| a * b).sum()
}

#[test]
fn dot_row_matches_dequantize_then_dot_for_every_format() {
    for (seed, format) in (1..).zip(formats()) {
        let blocks = 3;
        let row = pseudo_random_bytes(blocks * format.block_bytes, seed);
        let numel = blocks * format.block_size;
        let input: Vec<f32> = [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]
            .into_iter()
            .cycle()
            .take(numel)
            .collect();
        let values = format
            .dequantize(&row, &[numel])
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let expected = reference_dot(&values, &input);
        let got = format.dot_row(&row, &input);
        let tolerance = 1e-4 * values.iter().map(|v| v.abs()).sum::<f32>().max(1.0);
        assert!(
            (got - expected).abs() <= tolerance,
            "{}: {got} vs {expected}",
            format.name
        );
    }
}

#[test]
fn dequantize_row_matches_dequantize_for_every_format() {
    for (seed, format) in (1..).zip(formats()) {
        let row = pseudo_random_bytes(2 * format.block_bytes, seed);
        let numel = 2 * format.block_size;
        let expected = format
            .dequantize(&row, &[numel])
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let mut got = vec![0.0; numel];
        format.dequantize_row(&row, &mut got);
        assert_eq!(got, expected, "{}", format.name);
    }
}

#[test]
fn row_bytes_needs_whole_blocks() {
    let format = block_format(12).unwrap();
    assert_eq!(format.row_bytes(512), Some(288));
    assert_eq!(format.row_bytes(300), None);
    assert_eq!(WIDER_THAN_ITS_BLOCK.row_bytes(usize::MAX), None);
}
