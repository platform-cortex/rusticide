// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::{Q8_BLOCK, Q8Kind};

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
    dot_q8: |_, _| 0.0,
    dot_rows: None,
    activation: Q8Kind::Block,
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

/// Deterministic bytes over the whole `u8` range.
fn pseudo_random_bytes(len: usize, seed: u32) -> Vec<u8> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state.to_be_bytes()[0]
        })
        .collect()
}

/// Random blocks of `format` whose f16 scales are overwritten with finite
/// values, so every quant bit varies but no scale is infinite or NaN.
fn random_blocks(format: &BlockFormat, blocks: usize, seed: u32) -> Vec<u8> {
    let mut data = pseudo_random_bytes(blocks * format.block_bytes, seed);
    let scale_offsets: &[usize] = match format.name {
        "Q4_K" => &[0, 2],
        "Q6_K" => &[208],
        _ => &[0],
    };
    for (index, block) in data.chunks_exact_mut(format.block_bytes).enumerate() {
        for &offset in scale_offsets {
            let scale = 0.01 * f32::from(u8::try_from(index % 7 + 1).unwrap());
            block[offset..offset + 2].copy_from_slice(&half::f16::from_f32(scale).to_le_bytes());
        }
    }
    data
}

fn reference_dot(values: &[f32], input: &[f32]) -> f32 {
    values.iter().zip(input).map(|(a, b)| a * b).sum()
}

#[test]
fn dequantize_row_matches_dequantize_for_every_format() {
    for (seed, format) in (1..).zip(formats()) {
        let row = random_blocks(format, 2, seed);
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

#[test]
fn dot_row_q8_matches_the_dequantized_operands_for_every_format() {
    for (seed, format) in (1..).zip(formats()) {
        let blocks = 2;
        let row = random_blocks(format, blocks, seed);
        let numel = blocks * format.block_size;
        let raw: Vec<f32> = pseudo_random_bytes(numel, seed + 100)
            .into_iter()
            .map(|b| f32::from(b) / 128.0 - 1.0)
            .collect();
        let mut input = Q8Input::default();
        input.quantize(&raw, format.activation());
        let values = format
            .dequantize(&row, &[numel])
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let expected = reference_dot(&values, &input.dequantize());
        let got = format.dot_row_q8(&row, &input);
        let tolerance = 1e-4 * values.iter().map(|v| v.abs()).sum::<f32>().max(1.0);
        assert!(
            (got - expected).abs() <= tolerance,
            "{}: {got} vs {expected}",
            format.name
        );
    }
}

#[test]
fn every_format_block_spans_whole_activation_blocks() {
    assert!(
        formats()
            .iter()
            .all(|f| f.block_size.is_multiple_of(Q8_BLOCK))
    );
}

#[test]
fn dot_rows_q8_matches_one_dot_row_q8_per_input_bit_for_bit() {
    for (seed, format) in (1..).zip(formats()) {
        let blocks = 256 / format.block_size + 1;
        let row = random_blocks(format, blocks, seed);
        let inputs: Vec<Q8Input> = (0..3)
            .map(|token| {
                let raw: Vec<f32> = pseudo_random_bytes(blocks * format.block_size, seed + token)
                    .into_iter()
                    .map(|b| f32::from(b) / 64.0 - 2.0)
                    .collect();
                let mut input = Q8Input::default();
                input.quantize(&raw, format.activation());
                input
            })
            .collect();
        let expected: Vec<u32> = inputs
            .iter()
            .map(|input| format.dot_row_q8(&row, input).to_bits())
            .collect();
        let mut out = [f32::NAN; 3];
        format.dot_rows_q8(&row, &inputs, &mut out);
        assert_eq!(out.map(f32::to_bits), *expected, "{}", format.name);
        assert_ne!(out[0], out[1], "{}", format.name);
    }
}
