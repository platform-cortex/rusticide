// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_quantize::{BlockFormat, formats};

use super::QuantKernel;
use crate::test_support::{PAIRED_Q8_0, pseudo_random_values};
use crate::{Q8_BLOCK, Q8Input, Q8Kind, active};

fn kernel(format: &BlockFormat) -> &'static QuantKernel {
    active().quant_kernel(format.type_id()).unwrap()
}

fn quantized(len: usize, seed: u32, kind: Q8Kind) -> Q8Input {
    let mut input = Q8Input::default();
    input.quantize(&pseudo_random_values(len, seed), kind);
    input
}

fn reference_dot(values: &[f32], input: &[f32]) -> f32 {
    values.iter().zip(input).map(|(a, b)| a * b).sum()
}

#[test]
fn portable_has_a_kernel_for_every_block_format() {
    let found: Vec<(&str, u32, Q8Kind)> = formats()
        .iter()
        .map(|format| {
            let kernel = kernel(format);
            (format.name(), kernel.type_id(), kernel.activation())
        })
        .collect();
    assert_eq!(
        found,
        [
            ("Q4_0", 2, Q8Kind::Block),
            ("Q5_0", 6, Q8Kind::Block),
            ("Q4_K", 12, Q8Kind::SuperBlock),
            ("Q6_K", 14, Q8Kind::SuperBlock),
            ("Q8_0", 8, Q8Kind::Block),
        ]
    );
}

#[test]
fn an_unknown_type_id_has_no_kernel() {
    assert!(active().quant_kernel(99).is_none());
}

#[test]
fn dot_row_matches_the_dequantized_operands_for_every_format() {
    for (seed, format) in (1..).zip(formats()) {
        let kernel = kernel(format);
        let blocks = 2;
        let row = format.synthetic_blocks(blocks, seed).unwrap();
        let numel = blocks * format.block_size();
        let input = quantized(numel, seed + 100, kernel.activation());
        let mut values = vec![0.0; numel];
        format.dequantize_row(&row, &mut values);
        let expected = reference_dot(&values, &input.dequantize());
        let got = kernel.dot_row(&row, &input);
        let tolerance = 1e-4 * values.iter().map(|v| v.abs()).sum::<f32>().max(1.0);
        assert!(
            (got - expected).abs() <= tolerance,
            "{}: {got} vs {expected}",
            format.name()
        );
    }
}

#[test]
fn every_format_block_spans_whole_activation_blocks() {
    assert!(
        formats()
            .iter()
            .all(|f| f.block_size().is_multiple_of(Q8_BLOCK))
    );
}

#[test]
fn dot_rows_matches_one_dot_row_per_input_bit_for_bit() {
    for (seed, format) in (1..).zip(formats()) {
        let kernel = kernel(format);
        // Past one run of unpacked blocks for every format, ending mid-run.
        let blocks = 1280 / format.block_size() + 1;
        let row = format.synthetic_blocks(blocks, seed).unwrap();
        let inputs: Vec<Q8Input> = (0..3)
            .map(|token| {
                quantized(
                    blocks * format.block_size(),
                    seed + token,
                    kernel.activation(),
                )
            })
            .collect();
        let expected: Vec<u32> = inputs
            .iter()
            .map(|input| kernel.dot_row(&row, input).to_bits())
            .collect();
        let mut out = [f32::NAN; 3];
        kernel.dot_rows(&row, &inputs, &mut out);
        assert_eq!(out.map(f32::to_bits), *expected, "{}", format.name());
        assert_ne!(out[0], out[1], "{}", format.name());
    }
}

#[test]
fn batched_kernels_cover_q5_0_and_the_k_formats() {
    let batched: Vec<u32> = active()
        .quantized
        .iter()
        .filter(|kernel| kernel.dot_rows.is_some())
        .map(QuantKernel::type_id)
        .collect();
    assert_eq!(batched, [6, 12, 14]);
}

fn row_major_q8_0() -> &'static QuantKernel {
    crate::portable::ADAPTER.quant_kernel(8).unwrap()
}

#[test]
fn portable_kernels_read_rows_one_at_a_time() {
    for kernel in crate::portable::ADAPTER.quantized {
        assert!(kernel.layout.is_none());
    }
}

#[test]
fn read_row_restores_every_original_row() {
    let format = rusticide_quantize::block_format(8).unwrap();
    let original = format.synthetic_blocks(6, 7).unwrap();
    let row_bytes = original.len() / 6;
    let mut prepared = original.clone();
    assert!((PAIRED_Q8_0.layout.unwrap().prepare)(
        &mut prepared,
        row_bytes
    ));
    assert_ne!(prepared, original);
    for kernel in [row_major_q8_0(), &PAIRED_Q8_0] {
        let data = if kernel.layout.is_none() {
            &original
        } else {
            &prepared
        };
        for (index, expected) in original.chunks_exact(row_bytes).enumerate() {
            let mut row = vec![0; row_bytes];
            assert!(kernel.read_row(data, row_bytes, index, &mut row));
            assert_eq!(row, expected);
        }
    }
}

#[test]
fn read_row_rejects_a_missing_row_or_a_wrong_buffer() {
    let data = [1_u8, 2, 3, 4];
    let identity = row_major_q8_0();
    let mut row = [0_u8; 2];
    assert!(identity.read_row(&data, 2, 1, &mut row));
    assert_eq!(row, [3, 4]);
    assert!(!identity.read_row(&data, 2, 2, &mut row));
    assert!(!identity.read_row(&data, 2, usize::MAX, &mut row));
    assert!(!identity.read_row(&data, 1, 0, &mut row));
    assert!(!identity.read_row(&data, 0, 0, &mut []));
    assert_eq!(row, [3, 4]);
}

#[test]
fn the_paired_kernel_prepares_only_whole_pairs() {
    let prepare = PAIRED_Q8_0.layout.unwrap().prepare;
    let mut data = [1_u8, 2, 3, 4, 5, 6];
    assert!(!prepare(&mut data, 2));
    assert!(!prepare(&mut data, 0));
    assert_eq!(data, [1, 2, 3, 4, 5, 6]);
    assert!(prepare(&mut data, 3));
    assert_eq!(data, [4, 5, 6, 1, 2, 3]);
}

#[test]
fn visit_row_hands_over_the_original_row_of_either_layout() {
    let original = [1_u8, 2, 3, 4];
    let prepared = [3_u8, 4, 1, 2];
    for (kernel, data) in [(row_major_q8_0(), original), (&PAIRED_Q8_0, prepared)] {
        let mut seen = Vec::new();
        assert!(kernel.visit_row(&data, 2, 0, |row| seen.extend_from_slice(row)));
        assert_eq!(seen, [1, 2]);
        assert!(!kernel.visit_row(&data, 2, 2, |_| panic!("visited a missing row")));
    }
}

#[test]
fn dot_matrix_multiplies_every_row_of_either_layout() {
    let format = rusticide_quantize::block_format(8).unwrap();
    let original = format.synthetic_blocks(4, 9).unwrap();
    let row_bytes = original.len() / 4;
    let mut prepared = original.clone();
    assert!((PAIRED_Q8_0.layout.unwrap().prepare)(
        &mut prepared,
        row_bytes
    ));
    let inputs = [
        quantized(32, 1, Q8Kind::Block),
        quantized(32, 2, Q8Kind::Block),
    ];
    let expected: Vec<f32> = original
        .chunks_exact(row_bytes)
        .flat_map(|row| {
            inputs
                .iter()
                .map(|input| row_major_q8_0().dot_row(row, input))
        })
        .collect();
    for (kernel, data) in [(row_major_q8_0(), &original), (&PAIRED_Q8_0, &prepared)] {
        let mut out = [f32::NAN; 8];
        kernel.dot_matrix(data, row_bytes, &inputs, &mut out);
        assert_eq!(out.to_vec(), expected);
    }
}
