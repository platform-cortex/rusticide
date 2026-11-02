// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The NEON group kernels' common entry: [`Q8Input`]s viewed as the
//! kernel reads them (views built once per product, see `GroupViews`) and
//! handed to a `rusticide_simd` kernel a chunk of tokens at a time. A token
//! whose view is refused (it holds a `-128`) is computed by the format's
//! portable dot instead, and so is every token when no proof token is held.

use rusticide_simd::group::{GroupKernel, ROWS, TOKENS};

use crate::group_views::SchemeViews;
use crate::q8_input::Q8Input;
use crate::quant_kernel::QuantKernel;

/// One group against every input, into `out[row * inputs.len() + token]`.
/// `views` are those inputs' views; `format` supplies the row reader and
/// the dot for a refused token, or for every token when `proof` (the
/// instruction set's detected token) is `None`.
pub(super) fn dot_group<Proof: Copy, const UNIT: usize, View: Copy + Default>(
    format: &QuantKernel,
    proof: Option<Proof>,
    kernel: GroupKernel<Proof, UNIT, View>,
    views: &SchemeViews<'_, View>,
    group: &[u8],
    inputs: &[Q8Input],
    out: &mut [f32],
) {
    let Some(proof) = proof else {
        portable_tokens(format, group, inputs, 0..inputs.len(), out);
        return;
    };
    let (units, _) = group.as_chunks::<UNIT>();
    let tokens = inputs.len();
    for (first, chunk) in (0..).step_by(TOKENS).zip(views.views().chunks(TOKENS)) {
        let mut products = [[0.0; ROWS]; TOKENS];
        let count = chunk.len().min(TOKENS);
        kernel(proof, units, chunk, &mut products[..count]);
        for (row, slots) in (0..ROWS).zip(out.chunks_mut(tokens)) {
            for (slot, values) in slots.iter_mut().skip(first).zip(&products[..count]) {
                *slot = values[row];
            }
        }
    }
    if !views.refused().is_empty() {
        portable_tokens(format, group, inputs, views.refused().iter().copied(), out);
    }
}

/// Tokens `tokens` of every row of the group by the format's portable dot,
/// for activations the views refuse or a missing proof.
#[cold]
#[inline(never)]
fn portable_tokens(
    format: &QuantKernel,
    group: &[u8],
    inputs: &[Q8Input],
    tokens: impl Iterator<Item = usize>,
    out: &mut [f32],
) {
    let row_bytes = group.len() / ROWS;
    let stride = inputs.len();
    let mut row = vec![0; row_bytes];
    for (token, input) in tokens.filter_map(|token| Some((token, inputs.get(token)?))) {
        for (index, slots) in out.chunks_mut(stride).enumerate() {
            // A group the adapter prepared always reads: its length is whole
            // rows and `index < ROWS`.
            let read = format.read_row(group, row_bytes, index, &mut row);
            debug_assert!(read, "row {index} of a prepared group must read");
            slots[token] = format.dot_row(&row, input);
        }
    }
}
