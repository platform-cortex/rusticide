// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The NEON `Q4_K` kernel: [`Q8Input`]s viewed as whole super-blocks and
//! handed to `rusticide_simd` a chunk of tokens at a time.

use rusticide_simd::neon::q4_k::{Q8Superblocks, ROWS, TOKENS, UNIT, q4_k_group};

use super::q4_k::{prepare, read_row};
use crate::portable::int_dot;
use crate::q8_input::{Q8Input, Q8Kind};
use crate::quant_kernel::{Layout, QuantKernel};

pub(crate) const KERNEL: QuantKernel = QuantKernel {
    type_id: 12,
    activation: Q8Kind::SuperBlock,
    dot: int_dot::q4_k,
    dot_rows: Some(int_dot::q4_k_rows),
    layout: Some(Layout {
        group_rows: ROWS,
        prepare,
        read_row,
        dot_group,
    }),
};

fn superblocks(input: &Q8Input) -> Q8Superblocks<'_> {
    Q8Superblocks::new(
        input.scales.as_chunks().0,
        input.quants.as_chunks().0,
        input.sums.as_chunks().0,
    )
}

fn dot_group(group: &[u8], inputs: &[Q8Input], out: &mut [f32]) {
    let (units, _) = group.as_chunks::<UNIT>();
    let tokens = inputs.len();
    for (first, chunk) in (0..).step_by(TOKENS).zip(inputs.chunks(TOKENS)) {
        let mut views = [Q8Superblocks::default(); TOKENS];
        for (view, input) in views.iter_mut().zip(chunk) {
            *view = superblocks(input);
        }
        let mut products = [[0.0; ROWS]; TOKENS];
        let count = chunk.len().min(TOKENS);
        q4_k_group(units, &views[..count], &mut products[..count]);
        for (row, slots) in (0..ROWS).zip(out.chunks_mut(tokens)) {
            for (slot, values) in slots.iter_mut().skip(first).zip(&products[..count]) {
                *slot = values[row];
            }
        }
    }
}

#[cfg(test)]
#[path = "q4_k_dot_tests.rs"]
mod tests;
