// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The NEON group kernels' common entry: [`Q8Input`]s viewed as whole
//! super-blocks and handed to a `rusticide_simd` kernel a chunk of tokens
//! at a time.

use rusticide_simd::neon::group::{Q8Superblocks, ROWS, TOKENS};

use crate::q8_input::Q8Input;

/// A `rusticide_simd` kernel: one group's units against each input, the
/// eight rows side by side in the input's slot of the output.
pub(super) type GroupKernel<const UNIT: usize> =
    fn(group: &[[u8; UNIT]], inputs: &[Q8Superblocks<'_>], out: &mut [[f32; ROWS]]);

fn superblocks(input: &Q8Input) -> Q8Superblocks<'_> {
    Q8Superblocks::new(
        input.scales.as_chunks().0,
        input.quants.as_chunks().0,
        input.sums.as_chunks().0,
    )
}

/// One group against every input, into `out[row * inputs.len() + token]`.
pub(super) fn dot_group<const UNIT: usize>(
    kernel: GroupKernel<UNIT>,
    group: &[u8],
    inputs: &[Q8Input],
    out: &mut [f32],
) {
    let (units, _) = group.as_chunks::<UNIT>();
    let tokens = inputs.len();
    for (first, chunk) in (0..).step_by(TOKENS).zip(inputs.chunks(TOKENS)) {
        let mut views = [Q8Superblocks::default(); TOKENS];
        for (view, input) in views.iter_mut().zip(chunk) {
            *view = superblocks(input);
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
