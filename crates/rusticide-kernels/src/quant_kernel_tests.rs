// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_quantize::{BlockFormat, formats};

use super::QuantKernel;
use crate::test_support::pseudo_random_values;
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
        let row = format.synthetic_blocks(blocks, seed);
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
        let row = format.synthetic_blocks(blocks, seed);
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
