// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The NEON `Q8_0` kernel.

use rusticide_simd::Dotprod;
use rusticide_simd::group::ROWS;
use rusticide_simd::neon::q8_0::q8_0_group;

use super::group_dot;
use super::q8_0::{prepare, read_row};
use crate::group_views::GroupViews;
use crate::portable::int_dot;
use crate::q8_input::{Q8Input, Q8Kind};
use crate::quant_kernel::{Layout, QuantKernel};

pub(crate) const KERNEL: QuantKernel = QuantKernel {
    type_id: 8,
    activation: Q8Kind::Block,
    dot: int_dot::q8_0,
    dot_rows: None,
    layout: Some(Layout {
        group_rows: ROWS,
        prepare,
        read_row,
        dot_group,
    }),
};

fn dot_group(group: &[u8], inputs: &[Q8Input], views: &GroupViews<'_>, out: &mut [f32]) {
    let token = Dotprod::detect();
    group_dot::dot_group(
        &KERNEL,
        token,
        q8_0_group,
        views.blocks(),
        group,
        inputs,
        out,
    );
}

#[cfg(test)]
#[path = "q8_0_dot_tests.rs"]
mod tests;
