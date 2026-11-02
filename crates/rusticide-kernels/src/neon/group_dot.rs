// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The NEON group kernels' common entry: [`Q8Input`]s viewed as the
//! kernel reads them (views built once per product, see `GroupViews`) and
//! handed to a `rusticide_simd` kernel a chunk of tokens at a time. A token
//! whose view is refused (it holds a `-128`) is computed by the format's
//! portable dot instead, and so is every token when no proof token is held.

use rusticide_simd::group::{GroupKernel, ROWS, TOKENS};

use crate::q8_input::Q8Input;
use crate::quant_kernel::QuantKernel;

/// One group against every input, into `out[row * inputs.len() + token]`.
/// `views[token]` is that input's view; `format` supplies the row reader
/// and the dot for a token whose view is `None`, or for every token when
/// `proof` (the instruction set's detected token) is `None`.
pub(super) fn dot_group<Proof: Copy, const UNIT: usize, View: Copy + Default>(
    format: &QuantKernel,
    proof: Option<Proof>,
    kernel: GroupKernel<Proof, UNIT, View>,
    views: &[Option<View>],
    group: &[u8],
    inputs: &[Q8Input],
    out: &mut [f32],
) {
    let Some(proof) = proof else {
        let mut row = Vec::new();
        for (token, input) in inputs.iter().enumerate() {
            portable_token(format, group, input, token, &mut row, out);
        }
        return;
    };
    let (units, _) = group.as_chunks::<UNIT>();
    let tokens = inputs.len();
    let mut row = Vec::new();
    let chunks = inputs.chunks(TOKENS).zip(views.chunks(TOKENS));
    for (first, (chunk, found)) in (0..).step_by(TOKENS).zip(chunks) {
        let mut present = [View::default(); TOKENS];
        for (slot, view) in present.iter_mut().zip(found) {
            *slot = view.unwrap_or_default();
        }
        let mut products = [[0.0; ROWS]; TOKENS];
        let count = chunk.len().min(found.len()).min(TOKENS);
        kernel(proof, units, &present[..count], &mut products[..count]);
        for (row, slots) in (0..ROWS).zip(out.chunks_mut(tokens)) {
            for (slot, values) in slots.iter_mut().skip(first).zip(&products[..count]) {
                *slot = values[row];
            }
        }
        for (token, (input, _)) in (first..)
            .zip(chunk.iter().zip(found))
            .filter(|(_, (_, view))| view.is_none())
        {
            portable_token(format, group, input, token, &mut row, out);
        }
    }
}

/// Token `token` of every row of the group by the format's portable dot,
/// for an activation the views refuse. `row` is the caller's scratch.
#[cold]
#[inline(never)]
fn portable_token(
    format: &QuantKernel,
    group: &[u8],
    input: &Q8Input,
    token: usize,
    row: &mut Vec<u8>,
    out: &mut [f32],
) {
    let row_bytes = group.len() / ROWS;
    let tokens = out.len() / ROWS;
    row.resize(row_bytes, 0);
    for (index, slots) in out.chunks_mut(tokens).enumerate() {
        // A group the adapter prepared always reads: its length is whole
        // rows and `index < ROWS`.
        let read = format.read_row(group, row_bytes, index, row);
        debug_assert!(read, "row {index} of a prepared group must read");
        slots[token] = format.dot_row(row, input);
    }
}
