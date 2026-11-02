// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The NEON `Q6_K` kernel.

use rusticide_simd::group::ROWS;
use rusticide_simd::neon::q6_k::q6_k_group;

use super::group_dot;
use super::q6_k::{prepare, read_row};
use crate::group_views::GroupViews;
use crate::portable::int_dot;
use crate::q8_input::{Q8Input, Q8Kind};
use crate::quant_kernel::{Layout, QuantKernel};

pub(crate) const KERNEL: QuantKernel = QuantKernel {
    type_id: 14,
    activation: Q8Kind::SuperBlock,
    dot: int_dot::q6_k,
    dot_rows: Some(int_dot::q6_k_rows),
    layout: Some(Layout {
        group_rows: ROWS,
        prepare,
        read_row,
        dot_group,
    }),
};

fn dot_group(group: &[u8], inputs: &[Q8Input], views: &GroupViews<'_>, out: &mut [f32]) {
    group_dot::dot_group(&KERNEL, q6_k_group, views.superblocks(), group, inputs, out);
}

#[cfg(test)]
#[path = "q6_k_dot_tests.rs"]
mod tests;
