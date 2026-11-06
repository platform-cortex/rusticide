// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The AVX2 `Q4_K` kernel.

use rusticide_simd::Avx2;
use rusticide_simd::avx2::q4_k::q4_k_group;
use rusticide_simd::group::ROWS;

use crate::group_dot;
use crate::group_views::GroupViews;
use crate::portable;
use crate::q4_k_layout::{prepare, read_row};
use crate::q8_input::Q8Input;
use crate::quant_kernel::{Layout, QuantKernel};

pub(crate) const KERNEL: QuantKernel = QuantKernel {
    layout: Some(Layout {
        group_rows: ROWS,
        prepare,
        read_row,
        dot_group,
    }),
    ..portable::Q4_K
};

fn dot_group(group: &[u8], inputs: &[Q8Input], views: &GroupViews<'_>, out: &mut [f32]) {
    dot_group_with(super::proof(), group, inputs, views, out);
}

/// The group kernel with `proof`, or the portable kernel without one.
fn dot_group_with(
    proof: Option<Avx2>,
    group: &[u8],
    inputs: &[Q8Input],
    views: &GroupViews<'_>,
    out: &mut [f32],
) {
    group_dot::dot_group(
        &KERNEL,
        proof,
        q4_k_group,
        views.superblocks(),
        group,
        inputs,
        out,
    );
}

#[cfg(test)]
#[path = "q4_k_dot_tests.rs"]
mod tests;
