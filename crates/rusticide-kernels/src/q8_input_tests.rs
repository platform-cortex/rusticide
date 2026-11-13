// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use half::f16;

use super::*;
use crate::test_support::pseudo_random_values;

#[test]
fn quantize_scales_each_block_by_its_largest_magnitude() {
    let mut input = [0.0_f32; 64];
    input[0] = -254.0;
    input[1] = 128.0;
    input[32] = 0.5;
    let mut q8 = Q8Input::default();
    q8.quantize(&input, Q8Kind::Block);
    assert_eq!(q8.len(), 64);
    assert_eq!(q8.scales, [2.0, f16::from_f32(0.5 / 127.0).to_f32()]);
    assert_eq!(q8.quants[..2], [-127, 64]);
    assert_eq!(q8.quants[32], 127);
    assert_eq!(q8.sums, [-63, 127]);
}

#[test]
fn an_all_zero_block_has_zero_scale_and_quants() {
    let mut q8 = Q8Input::default();
    q8.quantize(&[0.0; 32], Q8Kind::Block);
    assert_eq!(q8.scales, [0.0]);
    assert!(q8.quants.iter().all(|&q| q == 0));
    assert_eq!(q8.dequantize(), [0.0; 32]);
}

#[test]
fn quantize_drops_a_partial_block_and_reuses_buffers() {
    let mut q8 = Q8Input::default();
    q8.quantize(&[1.0; 40], Q8Kind::Block);
    assert_eq!(q8.len(), 32);
    q8.quantize(&[], Q8Kind::SuperBlock);
    assert!(q8.is_empty());
}

#[test]
fn dequantize_reproduces_representable_values() {
    let mut input: Vec<f32> = (-16..16)
        .map(|v| f32::from(i8::try_from(v).unwrap()))
        .collect();
    input[31] = 127.0;
    let mut q8 = Q8Input::default();
    q8.quantize(&input, Q8Kind::Block);
    let decoded = q8.dequantize();
    assert!(
        decoded
            .iter()
            .zip(&input)
            .all(|(a, b)| (a - b).abs() < 1e-5)
    );
}

#[test]
fn a_super_block_shares_one_signed_scale() {
    let mut input = [0.0_f32; 256];
    input[0] = 1.0;
    input[40] = -2.0;
    input[255] = 0.5;
    let mut q8 = Q8Input::default();
    q8.quantize(&input, Q8Kind::SuperBlock);
    let scale = 1.0 / (-127.0 / -2.0);
    assert_eq!(q8.scales, [scale; 8]);
    assert_eq!(
        (q8.quants[0], q8.quants[40], q8.quants[255]),
        (64, -127, 32)
    );
    assert_eq!(q8.sums[0], 64);
    assert_eq!(q8.sums[1], -127);
}

#[test]
fn an_all_zero_super_block_has_zero_scale() {
    let mut q8 = Q8Input::default();
    q8.quantize(&[0.0; 256], Q8Kind::SuperBlock);
    assert_eq!(q8.scales, [0.0; 8]);
}

/// A row of `Q8_BLOCK` elements starting with `head`, zeros after.
fn row(head: &[f32]) -> Vec<f32> {
    let mut values = vec![0.0; Q8_BLOCK];
    values[..head.len()].copy_from_slice(head);
    values
}

fn quantized_row(values: &[f32]) -> Q8Input {
    let mut q8 = Q8Input::default();
    q8.quantize(values, Q8Kind::Row);
    q8
}

/// ggml's `nearest_int`: the `12582912.f` trick, rounding half to even.
fn nearest_int(value: f32) -> i32 {
    let bits = (value + 12_582_912.0).to_bits().cast_signed();
    (bits & 0x007f_ffff) - 0x0040_0000
}

/// bitnet.cpp's legacy `quantize_row_i8_s`, C's `MAX` macro included.
fn reference_row(values: &[f32]) -> (f32, Vec<i8>) {
    let mut max = 1e-5_f64;
    for value in values {
        let magnitude = f64::from(value.abs());
        max = if max > magnitude { max } else { magnitude };
    }
    #[expect(clippy::cast_possible_truncation, reason = "the reference's (float)")]
    let scale = (127.0 / max) as f32;
    let quants = values
        .iter()
        .map(|&v| i8::try_from(nearest_int(v * scale).clamp(-128, 127)).unwrap())
        .collect();
    (scale, quants)
}

#[test]
fn row_scale_follows_the_legacy_formula() {
    let mut values = row(&[0.5, -2.0, 1.0]);
    values.extend(row(&[]));
    let q8 = quantized_row(&values);
    #[expect(clippy::cast_possible_truncation, reason = "the reference's (float)")]
    let scale = (127.0_f64 / 2.0) as f32;
    assert_eq!(q8.scales, [scale; 2]);
    assert_eq!(q8.quants[..3], [32, -127, 64]);
    assert_eq!(q8.sums, [-31, 0]);
}

#[test]
fn an_all_zero_row_quantizes_to_zeros() {
    let q8 = quantized_row(&row(&[]));
    #[expect(clippy::cast_possible_truncation, reason = "the reference's (float)")]
    let scale = (127.0 / 1e-5_f64) as f32;
    assert_eq!(q8.scales, [scale]);
    assert!(q8.quants.iter().all(|&q| q == 0));
    assert_eq!(q8.sums, [0]);
}

#[test]
fn halfway_rounds_to_even() {
    let values = row(&[127.0, 2.5, 3.5, -2.5, -3.5]);
    let q8 = quantized_row(&values);
    assert_eq!(q8.scales, [1.0]);
    assert_eq!(q8.quants[..5], [127, 2, 4, -2, -4]);
    assert_eq!(reference_row(&values).1, q8.quants);
}

#[test]
fn row_clamps_to_i8() {
    let values = row(&[-4.0, f32::NAN, 1.0]);
    let q8 = quantized_row(&values);
    assert_eq!(q8.scales, [127.0]);
    assert_eq!(q8.quants[..3], [-128, 0, 127]);
    assert_eq!(reference_row(&values), (127.0, q8.quants));
}

#[test]
fn row_sums_are_the_block_quant_sums() {
    let values: Vec<f32> = pseudo_random_values(256, 3)
        .iter()
        .map(|v| v * 3.7)
        .collect();
    let q8 = quantized_row(&values);
    let sums: Vec<i32> = q8
        .quants
        .chunks(Q8_BLOCK)
        .map(|block| block.iter().map(|&q| i32::from(q)).sum())
        .collect();
    assert_eq!(q8.sums, sums);
    let (scale, quants) = reference_row(&values);
    assert_eq!((q8.scales, q8.quants), (vec![scale; 8], quants));
}

#[test]
fn dequantize_divides_a_row_by_its_scale() {
    let q8 = quantized_row(&row(&[0.5, -2.0, 1.0]));
    let scale = q8.scales[0];
    let expected: Vec<f32> = q8.quants.iter().map(|&q| f32::from(q) / scale).collect();
    assert_eq!(q8.dequantize(), expected);
}
