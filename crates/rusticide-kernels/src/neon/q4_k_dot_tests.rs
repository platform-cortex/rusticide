// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_quantize::block_format;

use crate::neon::ADAPTER;
use crate::portable::{self, int_dot};
use crate::q8_input::{Q8Input, Q8Kind};
use crate::quant_kernel::QuantKernel;
use crate::test_support::pseudo_random_values;

const ROWS: usize = 16;
const COLS: usize = 1024;

fn weights(seed: u32) -> (Vec<u8>, usize) {
    let format = block_format(12).unwrap();
    let data = format.synthetic_blocks(ROWS * COLS / 256, seed).unwrap();
    (data, format.row_bytes(COLS).unwrap())
}

fn inputs(tokens: usize, seed: u32) -> Vec<Q8Input> {
    (0..tokens)
        .map(|token| {
            let mut input = Q8Input::default();
            let values = pseudo_random_values(COLS, seed + u32::try_from(token).unwrap());
            input.quantize(&values, Q8Kind::SuperBlock);
            input
        })
        .collect()
}

fn neon_q4_k() -> &'static QuantKernel {
    ADAPTER.quant_kernel(12).unwrap()
}

fn neon_products(original: &[u8], row_bytes: usize, inputs: &[Q8Input]) -> Vec<f32> {
    let mut data = original.to_vec();
    let kernel = ADAPTER.prepare_matrix(12, &mut data, row_bytes).unwrap();
    assert!(std::ptr::eq(kernel, neon_q4_k()));
    let mut out = vec![f32::NAN; ROWS * inputs.len()];
    kernel.compute_rows(&data, row_bytes, 0, inputs, &mut out);
    out
}

#[test]
fn neon_q4_k_matches_portable() {
    let (original, row_bytes) = weights(21);
    for tokens in [1, 2, 7, 64] {
        let inputs = inputs(tokens, 40);
        let got = neon_products(&original, row_bytes, &inputs);
        for (row, values) in original
            .chunks_exact(row_bytes)
            .zip(got.chunks_exact(tokens))
        {
            let mut expected = vec![0.0; tokens];
            int_dot::q4_k_rows(row, &inputs, &mut expected);
            for (&got, &expected) in values.iter().zip(&expected) {
                let tolerance = (1e-5 * expected.abs()).max(1e-6);
                assert!(
                    (got - expected).abs() <= tolerance,
                    "{got} vs {expected}, {tokens} tokens"
                );
            }
        }
    }
}

#[test]
fn a_batch_is_bit_identical_to_its_tokens_one_at_a_time() {
    let (original, row_bytes) = weights(22);
    let tokens = 19;
    let inputs = inputs(tokens, 50);
    let batch = neon_products(&original, row_bytes, &inputs);
    for (token, input) in inputs.iter().enumerate() {
        let alone = neon_products(&original, row_bytes, std::slice::from_ref(input));
        let column: Vec<u32> = batch
            .iter()
            .skip(token)
            .step_by(tokens)
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(
            column,
            alone.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "token {token}"
        );
    }
}

#[test]
fn no_inputs_leave_the_output_untouched() {
    let (mut data, row_bytes) = weights(23);
    assert!((neon_q4_k().layout.unwrap().prepare)(&mut data, row_bytes));
    let mut out = [1.5; ROWS];
    neon_q4_k().compute_rows(&data, row_bytes, 0, &[], &mut out);
    assert_eq!(out, [1.5; ROWS]);
}

#[test]
fn the_adapter_runs_q4_k_on_neon_in_groups_of_eight() {
    assert_eq!(neon_q4_k().layout.map(|layout| layout.group_rows), Some(8));
    let portable_types: Vec<u32> = portable::ADAPTER
        .quantized
        .iter()
        .map(|k| k.type_id)
        .collect();
    let neon_types: Vec<u32> = ADAPTER.quantized.iter().map(|k| k.type_id).collect();
    assert_eq!(neon_types, portable_types);
    let (original, row_bytes) = weights(24);
    let mut data = original[..3 * row_bytes].to_vec();
    let kernel = ADAPTER.prepare_matrix(12, &mut data, row_bytes).unwrap();
    assert!(std::ptr::eq(
        kernel,
        portable::ADAPTER.quant_kernel(12).unwrap()
    ));
}
