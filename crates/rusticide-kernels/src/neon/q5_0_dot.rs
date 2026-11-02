// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The NEON `Q5_0` kernel.

use rusticide_simd::Dotprod;
use rusticide_simd::group::ROWS;
use rusticide_simd::neon::q5_0::q5_0_group;

use super::group_dot;
use super::q5_0::{prepare, read_row};
use crate::group_views::GroupViews;
use crate::portable::int_dot;
use crate::q8_input::{Q8Input, Q8Kind};
use crate::quant_kernel::{Layout, QuantKernel};

pub(crate) const KERNEL: QuantKernel = QuantKernel {
    type_id: 6,
    activation: Q8Kind::Block,
    dot: int_dot::q5_0,
    dot_rows: Some(int_dot::q5_0_rows),
    layout: Some(Layout {
        group_rows: ROWS,
        prepare,
        read_row,
        dot_group,
    }),
};

fn dot_group(group: &[u8], inputs: &[Q8Input], views: &GroupViews<'_>, out: &mut [f32]) {
    dot_group_with(super::proof(), group, inputs, views, out);
}

/// The group kernel with `proof`, or the portable kernel without one.
fn dot_group_with(
    proof: Option<Dotprod>,
    group: &[u8],
    inputs: &[Q8Input],
    views: &GroupViews<'_>,
    out: &mut [f32],
) {
    group_dot::dot_group(
        &KERNEL,
        proof,
        q5_0_group,
        views.blocks(),
        group,
        inputs,
        out,
    );
}

#[cfg(test)]
#[path = "q5_0_dot_tests.rs"]
mod tests;
