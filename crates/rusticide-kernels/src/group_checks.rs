// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Checks every SIMD group kernel must pass, by GGML type id, against
//! this architecture's SIMD adapter.

use rusticide_quantize::block_format;
use rusticide_simd::group::TOKENS;

#[cfg(target_arch = "x86_64")]
use crate::avx2::{ADAPTER, proof};
use crate::group_views::{GroupViews, visit_views};
#[cfg(target_arch = "aarch64")]
use crate::neon::{ADAPTER, proof};
use crate::portable;
use crate::q8_input::Q8Input;
use crate::quant_kernel::{Layout, QuantKernel};
use crate::test_support::pseudo_random_values;

const ROWS: usize = 16;
const COLS: usize = 1024;

/// A format's group kernel wrapper with its proof withheld.
pub(crate) type Unproved = fn(&[u8], &[Q8Input], &GroupViews<'_>, &mut [f32]);

fn kernel(type_id: u32) -> &'static QuantKernel {
    ADAPTER.quant_kernel(type_id).unwrap()
}

fn layout(type_id: u32) -> Layout {
    kernel(type_id).layout.unwrap()
}

/// `ROWS` rows of `blocks` random `block`-byte blocks.
fn random_rows(block: usize, blocks: usize, seed: u64) -> (Vec<u8>, usize) {
    let mut rng = fastrand::Rng::with_seed(seed);
    let row_bytes = blocks * block;
    let data = (0..ROWS * row_bytes).map(|_| rng.u8(..)).collect();
    (data, row_bytes)
}

fn weights(type_id: u32, seed: u32) -> (Vec<u8>, usize) {
    let format = block_format(type_id).unwrap();
    let data = format
        .synthetic_blocks(ROWS * COLS / format.block_size(), seed)
        .unwrap();
    (data, format.row_bytes(COLS).unwrap())
}

/// `tokens` activations quantized as the kernel for `type_id` pairs them.
fn inputs(type_id: u32, tokens: usize, seed: u32) -> Vec<Q8Input> {
    let kind = kernel(type_id).activation;
    (0..tokens)
        .map(|token| {
            let mut input = Q8Input::default();
            let values = pseudo_random_values(COLS, seed + u32::try_from(token).unwrap());
            input.quantize(&values, kind);
            input
        })
        .collect()
}

/// The products through the adapter, after detection stored the proof the
/// wrappers read, so the SIMD kernels (not the portable fallback) run.
fn simd_products(type_id: u32, original: &[u8], row_bytes: usize, inputs: &[Q8Input]) -> Vec<f32> {
    assert!(
        crate::adapter::available()
            .iter()
            .any(|adapter| adapter.name() == ADAPTER.name())
    );
    assert!(proof().is_some());
    let mut data = original.to_vec();
    let prepared = ADAPTER
        .prepare_matrix(type_id, &mut data, row_bytes)
        .unwrap();
    assert!(std::ptr::eq(prepared, kernel(type_id)));
    assert!(prepared.layout.is_some());
    let mut out = vec![f32::NAN; ROWS * inputs.len()];
    prepared.dot_matrix(&data, row_bytes, inputs, &mut out);
    out
}

pub(crate) fn prepare_then_read_row_is_identity(type_id: u32, block: usize) {
    let Layout {
        prepare, read_row, ..
    } = layout(type_id);
    for blocks in [1, 3] {
        let (original, row_bytes) = random_rows(block, blocks, 5);
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

/// Row 3's second block, after preparing two-block rows, and the unit
/// that holds it.
pub(crate) fn second_unit_and_block(type_id: u32, block: usize) -> (Vec<u8>, Vec<u8>) {
    let (original, row_bytes) = random_rows(block, 2, 9);
    let mut data = original.clone();
    assert!((layout(type_id).prepare)(&mut data, row_bytes));
    let unit = data[8 * block..16 * block].to_vec();
    let start = 3 * row_bytes + block;
    (unit, original[start..start + block].to_vec())
}

pub(crate) fn prepare_leaves_partial_groups_or_blocks_untouched(type_id: u32, block: usize) {
    let prepare = layout(type_id).prepare;
    let (original, row_bytes) = random_rows(block, 1, 3);
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
    let overflowing = block * (usize::MAX / block / 4);
    assert!(!prepare(&mut [0; 8], overflowing));
    assert!(prepare(&mut [], row_bytes));
}

pub(crate) fn read_row_rejects_a_missing_row_or_a_wrong_buffer(type_id: u32, block: usize) {
    let Layout {
        prepare, read_row, ..
    } = layout(type_id);
    let (mut data, row_bytes) = random_rows(block, 2, 4);
    assert!(prepare(&mut data, row_bytes));
    let mut row = vec![7; row_bytes];
    assert!(!read_row(&data, row_bytes, ROWS, &mut row));
    assert!(!read_row(&data, row_bytes, usize::MAX, &mut row));
    assert!(!read_row(&data, row_bytes, 0, &mut row[1..]));
    assert!(!read_row(&data, 0, 0, &mut []));
    assert!(!read_row(&data, block + 1, 0, &mut row[..=block]));
    assert_eq!(row, vec![7; row_bytes]);
}

/// More than a chunk of tokens, the last chunk partial, included, against
/// the portable kernel's batched rows.
pub(crate) fn simd_matches_portable(type_id: u32) {
    let portable = portable::ADAPTER.quant_kernel(type_id).unwrap();
    let (original, row_bytes) = weights(type_id, 21);
    for tokens in [1, 2, 7, 19, 64] {
        let inputs = inputs(type_id, tokens, 40);
        let got = simd_products(type_id, &original, row_bytes, &inputs);
        for (row, values) in original
            .chunks_exact(row_bytes)
            .zip(got.chunks_exact(tokens))
        {
            let mut expected = vec![0.0; tokens];
            portable.dot_rows(row, &inputs, &mut expected);
            for (&got, &expected) in values.iter().zip(&expected) {
                assert_eq!(
                    got.to_bits(),
                    expected.to_bits(),
                    "{got} vs {expected}, {tokens} tokens"
                );
            }
        }
    }
}

pub(crate) fn a_batch_is_bit_identical_to_its_tokens_one_at_a_time(type_id: u32) {
    let (original, row_bytes) = weights(type_id, 22);
    let tokens = 19;
    let inputs = inputs(type_id, tokens, 50);
    let batch = simd_products(type_id, &original, row_bytes, &inputs);
    for (token, input) in inputs.iter().enumerate() {
        let alone = simd_products(type_id, &original, row_bytes, std::slice::from_ref(input));
        let column: Vec<u32> = batch
            .iter()
            .skip(token)
            .step_by(tokens)
            .map(|v| v.to_bits())
            .collect();
        let alone: Vec<u32> = alone.iter().map(|v| v.to_bits()).collect();
        assert_eq!(column, alone, "token {token}");
    }
}

pub(crate) fn no_inputs_leave_the_output_untouched(type_id: u32) {
    let (mut data, row_bytes) = weights(type_id, 23);
    assert!((layout(type_id).prepare)(&mut data, row_bytes));
    let mut out = [1.5; ROWS];
    kernel(type_id).dot_matrix(&data, row_bytes, &[], &mut out);
    assert_eq!(out, [1.5; ROWS]);
}

/// The kernel groups eight rows. A matrix that is not whole groups (fewer
/// rows, or a whole group and a remainder) or has zero-byte rows is not
/// grouped at all: the whole matrix stays row-major and the portable
/// kernel reads it.
pub(crate) fn runs_in_groups_of_eight(type_id: u32) {
    assert_eq!(layout(type_id).group_rows, 8);
    let portable_types: Vec<u32> = portable::ADAPTER
        .quantized
        .iter()
        .map(|k| k.type_id)
        .collect();
    let simd_types: Vec<u32> = ADAPTER.quantized.iter().map(|k| k.type_id).collect();
    assert_eq!(simd_types, portable_types);
    let (original, row_bytes) = weights(type_id, 24);
    let fallback = portable::ADAPTER.quant_kernel(type_id).unwrap();
    let inputs = inputs(type_id, 3, 80);
    for rows in [3, 11] {
        let original = &original[..rows * row_bytes];
        let mut data = original.to_vec();
        let prepared = ADAPTER
            .prepare_matrix(type_id, &mut data, row_bytes)
            .unwrap();
        assert!(std::ptr::eq(prepared, fallback), "{rows} rows");
        assert_eq!(data, original, "{rows} rows");
        let mut got = vec![f32::NAN; rows * inputs.len()];
        prepared.dot_matrix(&data, row_bytes, &inputs, &mut got);
        assert_portable(type_id, original, row_bytes, &inputs, &got);
    }
    let zero_byte_rows = ADAPTER.prepare_matrix(type_id, &mut [], 0).unwrap();
    assert!(std::ptr::eq(zero_byte_rows, fallback));
}

/// Tokens in the first, second and last (partial) chunk carry a `-128`,
/// which the group views refuse: every value still equals the portable
/// kernel's.
pub(crate) fn a_minus_128_activation_matches_portable(type_id: u32) {
    let (original, row_bytes) = weights(type_id, 25);
    let mut inputs = inputs(type_id, 2 * TOKENS + 3, 60);
    for (token, element) in [(3, 5), (TOKENS + 4, 700), (2 * TOKENS + 2, 9)] {
        let input = &mut inputs[token];
        let old = i32::from(input.quants[element]);
        input.quants[element] = i8::MIN;
        input.sums[element / 32] += i32::from(i8::MIN) - old;
    }
    let got = simd_products(type_id, &original, row_bytes, &inputs);
    assert_portable(type_id, &original, row_bytes, &inputs, &got);
}

/// `got` holds, row by row, each of `original`'s rows against every input
/// exactly as the portable kernel computes it.
fn assert_portable(
    type_id: u32,
    original: &[u8],
    row_bytes: usize,
    inputs: &[Q8Input],
    got: &[f32],
) {
    let portable = portable::ADAPTER.quant_kernel(type_id).unwrap();
    for (row, values) in original
        .chunks_exact(row_bytes)
        .zip(got.chunks_exact(inputs.len()))
    {
        for (value, input) in values.iter().zip(inputs) {
            assert_eq!(value.to_bits(), portable.dot_row(row, input).to_bits());
        }
    }
}

/// A group kernel wrapper handed no proof computes every value with the
/// portable kernel instead, never the SIMD one.
pub(crate) fn wrapper_without_token_matches_portable(type_id: u32, unproved: Unproved) {
    let (original, row_bytes) = weights(type_id, 26);
    let mut data = original.clone();
    assert!((layout(type_id).prepare)(&mut data, row_bytes));
    let inputs = inputs(type_id, TOKENS + 3, 70);
    let mut got = vec![f32::NAN; ROWS * inputs.len()];
    let group_rows = layout(type_id).group_rows;
    visit_views(&inputs, kernel(type_id).activation, |views| {
        let groups = data.chunks_exact(group_rows * row_bytes);
        for (group, out) in groups.zip(got.chunks_exact_mut(group_rows * inputs.len())) {
            unproved(group, &inputs, views, out);
        }
    });
    assert_portable(type_id, &original, row_bytes, &inputs, &got);
}
