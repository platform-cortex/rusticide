// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The NEON `Q4_K` kernel.

use rusticide_simd::neon::group::ROWS;
use rusticide_simd::neon::q4_k::q4_k_group;

use super::group_dot;
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

fn dot_group(group: &[u8], inputs: &[Q8Input], out: &mut [f32]) {
    group_dot::dot_group(q4_k_group, group_dot::superblocks, group, inputs, out);
}

#[cfg(test)]
#[path = "q4_k_dot_tests.rs"]
mod tests;
