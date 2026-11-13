// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{dot_ternary_matrix, ternary_dot};
use crate::test_support::{naive_ternary_dot, pseudo_random_values, ternary_rows};
use crate::{Q8Input, Q8Kind};

fn quantized(len: usize, seed: u32) -> Q8Input {
    let mut input = Q8Input::default();
    input.quantize(&pseudo_random_values(len, seed), Q8Kind::Row);
    input
}

#[test]
fn the_dot_equals_a_naive_integer_dot_on_random_rows() {
    for seed in 1..6 {
        let row = ternary_rows(5, seed);
        let codes: Vec<u8> = row.iter().map(|byte| byte >> 6).collect();
        assert!((0..4).all(|code| codes.contains(&code)));
        let input = quantized(640, seed + 50);
        assert_eq!(ternary_dot(&row, &input), naive_ternary_dot(&row, &input));
    }
}

#[test]
fn code_three_counts_as_plus_two() {
    let input = quantized(128, 7);
    let total: i32 = input.quants.iter().map(|&q| i32::from(q)).sum();
    assert_ne!(total, 0);
    assert_eq!(ternary_dot(&[0xFF; 32], &input), 2 * total);
    assert_eq!(ternary_dot(&[0x55; 32], &input), 0);
}

#[test]
fn the_shorter_operand_bounds_the_dot() {
    let row = ternary_rows(3, 9);
    let input = quantized(256, 10);
    assert_eq!(
        ternary_dot(&row, &input),
        naive_ternary_dot(&row[..64], &input)
    );
    assert_eq!(
        ternary_dot(&row[..40], &input),
        naive_ternary_dot(&row[..32], &input)
    );
}

#[test]
fn a_hostile_width_wraps_rather_than_panics() {
    let blocks = (1 << 31) / (254 * 128) + 1;
    let input = Q8Input {
        kind: Q8Kind::Row,
        scales: vec![1.0; blocks * 4],
        quants: vec![127; blocks * 128],
        sums: vec![127 * 32; blocks * 4],
    };
    let exact = 254_i64 * 128 * i64::try_from(blocks).unwrap();
    assert!(exact > i64::from(i32::MAX));
    let wrapped = i32::try_from(exact - (1_i64 << 32)).unwrap();
    assert_eq!(ternary_dot(&vec![0xFF; blocks * 32], &input), wrapped);
}

/// `(dot as f32) / s * scale`, in the legacy order.
#[expect(
    clippy::cast_precision_loss,
    reason = "the test dots are far below 2^24"
)]
fn legacy(row: &[u8], input: &Q8Input, scale: f32) -> f32 {
    naive_ternary_dot(row, input) as f32 / input.scales[0] * scale
}

#[test]
fn the_matrix_scales_each_row_and_token_in_the_legacy_order() {
    let data = ternary_rows(6, 3);
    let inputs = [quantized(256, 4), quantized(256, 5)];
    let mut out = [f32::NAN; 6];
    dot_ternary_matrix(&data, 64, 1.25, &inputs, &mut out);
    let expected: Vec<u32> = data
        .as_chunks::<64>()
        .0
        .iter()
        .flat_map(|row| inputs.iter().map(move |input| legacy(row, input, 1.25)))
        .map(f32::to_bits)
        .collect();
    assert_eq!(out.map(f32::to_bits), *expected);
    assert_ne!(out[0], out[1]);
}

#[test]
fn a_zero_row_is_exactly_zero() {
    let mut out = [f32::NAN; 2];
    dot_ternary_matrix(&[0x55; 64], 32, 3.0, &[quantized(128, 8)], &mut out);
    assert_eq!(out.map(f32::to_bits), [0, 0]);
}

#[test]
fn an_empty_input_is_zero_and_a_zero_width_writes_nothing() {
    let mut out = [f32::NAN; 1];
    dot_ternary_matrix(&[0xFF; 32], 32, 1.0, &[Q8Input::default()], &mut out);
    assert_eq!(out[0].to_bits(), 0);
    let mut out = [f32::NAN; 1];
    dot_ternary_matrix(&[0xFF; 32], 0, 1.0, &[quantized(128, 1)], &mut out);
    assert!(out[0].is_nan());
}
