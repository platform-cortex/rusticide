// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The NEON group kernels' common entry: [`Q8Input`]s viewed as the
//! kernel reads them and handed to a `rusticide_simd` kernel a chunk of
//! tokens at a time.

use rusticide_simd::neon::group::{GroupKernel, Q8Blocks, Q8Superblocks, ROWS, TOKENS};

use crate::q8_input::Q8Input;

pub(super) fn superblocks(input: &Q8Input) -> Q8Superblocks<'_> {
    Q8Superblocks::new(
        input.scales.as_chunks().0,
        input.quants.as_chunks().0,
        input.sums.as_chunks().0,
    )
}

pub(super) fn blocks(input: &Q8Input) -> Q8Blocks<'_> {
    Q8Blocks::new(&input.scales, input.quants.as_chunks().0)
}

/// One group against every input, into `out[row * inputs.len() + token]`.
pub(super) fn dot_group<'a, const UNIT: usize, View: Copy + Default>(
    kernel: GroupKernel<UNIT, View>,
    view: fn(&'a Q8Input) -> View,
    group: &[u8],
    inputs: &'a [Q8Input],
    out: &mut [f32],
) {
    let (units, _) = group.as_chunks::<UNIT>();
    let tokens = inputs.len();
    for (first, chunk) in (0..).step_by(TOKENS).zip(inputs.chunks(TOKENS)) {
        let mut views = [View::default(); TOKENS];
        for (slot, input) in views.iter_mut().zip(chunk) {
            *slot = view(input);
        }
        let mut products = [[0.0; ROWS]; TOKENS];
        let count = chunk.len().min(TOKENS);
        kernel(units, &views[..count], &mut products[..count]);
        for (row, slots) in (0..ROWS).zip(out.chunks_mut(tokens)) {
            for (slot, values) in slots.iter_mut().skip(first).zip(&products[..count]) {
                *slot = values[row];
            }
        }
    }
}
