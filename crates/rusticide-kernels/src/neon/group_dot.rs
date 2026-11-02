// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The NEON group kernels' common entry: [`Q8Input`]s viewed as the
//! kernel reads them and handed to a `rusticide_simd` kernel a chunk of
//! tokens at a time. A token whose view is refused (it holds a `-128`) is
//! computed by the format's portable dot instead.

use rusticide_simd::group::{GroupKernel, Q8Blocks, Q8Superblocks, ROWS, TOKENS};

use crate::q8_input::Q8Input;
use crate::quant_kernel::QuantKernel;

pub(super) fn superblocks(input: &Q8Input) -> Option<Q8Superblocks<'_>> {
    Q8Superblocks::new(
        input.scales.as_chunks().0,
        input.quants.as_chunks().0,
        input.sums.as_chunks().0,
    )
}

pub(super) fn blocks(input: &Q8Input) -> Option<Q8Blocks<'_>> {
    Q8Blocks::new(&input.scales, input.quants.as_chunks().0)
}

/// One group against every input, into `out[row * inputs.len() + token]`.
/// `format` supplies the row reader and the dot for refused tokens.
pub(super) fn dot_group<'a, const UNIT: usize, View: Copy + Default>(
    format: &QuantKernel,
    kernel: GroupKernel<UNIT, View>,
    view: fn(&'a Q8Input) -> Option<View>,
    group: &[u8],
    inputs: &'a [Q8Input],
    out: &mut [f32],
) {
    let (units, _) = group.as_chunks::<UNIT>();
    let tokens = inputs.len();
    for (first, chunk) in (0..).step_by(TOKENS).zip(inputs.chunks(TOKENS)) {
        let mut views = [View::default(); TOKENS];
        let mut refused = [false; TOKENS];
        for ((slot, refuse), input) in views.iter_mut().zip(&mut refused).zip(chunk) {
            let found = view(input);
            *refuse = found.is_none();
            *slot = found.unwrap_or_default();
        }
        let mut products = [[0.0; ROWS]; TOKENS];
        let count = chunk.len().min(TOKENS);
        kernel(units, &views[..count], &mut products[..count]);
        for (row, slots) in (0..ROWS).zip(out.chunks_mut(tokens)) {
            for (slot, values) in slots.iter_mut().skip(first).zip(&products[..count]) {
                *slot = values[row];
            }
        }
        for ((token, input), _) in (first..)
            .zip(chunk)
            .zip(&refused)
            .filter(|(_, refuse)| **refuse)
        {
            portable_token(format, group, input, token, out);
        }
    }
}

/// Token `token` of every row of the group by the format's portable dot,
/// for an activation the views refuse.
fn portable_token(
    format: &QuantKernel,
    group: &[u8],
    input: &Q8Input,
    token: usize,
    out: &mut [f32],
) {
    let row_bytes = group.len() / ROWS;
    let tokens = out.len() / ROWS;
    let mut row = vec![0; row_bytes];
    for (index, slots) in out.chunks_mut(tokens).enumerate() {
        if format.read_row(group, row_bytes, index, &mut row) {
            slots[token] = format.dot_row(&row, input);
        }
    }
}
