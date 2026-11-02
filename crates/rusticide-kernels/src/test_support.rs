// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Test vehicles for this crate's tests and, behind the `test-kernels`
//! feature, for the tests of crates that load weights through an adapter.

use crate::adapter::Adapter;
use crate::group_views::GroupViews;
use crate::portable::{self, int_dot};
use crate::q8_input::{Q8Input, Q8Kind};
use crate::quant_kernel::{Layout, QuantKernel, read_row_major};

/// A `Q8_0` kernel whose layout groups rows in pairs, each pair stored
/// swapped: the smallest layout a row-major reader gets wrong.
pub static PAIRED_Q8_0: QuantKernel = QuantKernel {
    type_id: 8,
    activation: Q8Kind::Block,
    dot: int_dot::q8_0,
    dot_rows: None,
    layout: Some(Layout {
        group_rows: 2,
        prepare: swap_pairs,
        read_row: read_swapped,
        dot_group: dot_swapped,
    }),
};

/// The portable adapter with [`PAIRED_Q8_0`] as its only quantized kernel.
pub static PAIRED: Adapter = Adapter {
    name: "paired",
    quantized: std::slice::from_ref(&PAIRED_Q8_0),
    ..portable::ADAPTER
};

fn swap_pairs(data: &mut [u8], row_bytes: usize) -> bool {
    let pair = row_bytes.saturating_mul(2);
    if pair == 0 || !data.len().is_multiple_of(pair) {
        return false;
    }
    for rows in data.chunks_exact_mut(pair) {
        let (first, second) = rows.split_at_mut(row_bytes);
        first.swap_with_slice(second);
    }
    true
}

fn read_swapped(data: &[u8], row_bytes: usize, index: usize, out: &mut [u8]) -> bool {
    read_row_major(data, row_bytes, index ^ 1, out)
}

fn dot_swapped(group: &[u8], inputs: &[Q8Input], _views: &GroupViews<'_>, out: &mut [f32]) {
    let (second, first) = group.split_at(group.len() / 2);
    for (slots, row) in out.chunks_mut(inputs.len().max(1)).zip([first, second]) {
        for (slot, input) in slots.iter_mut().zip(inputs) {
            *slot = int_dot::q8_0(row, input);
        }
    }
}

/// Asserts two slices match element-wise within a small tolerance.
#[cfg(test)]
pub(crate) fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < 1e-6, "{actual:?} != {expected:?}");
    }
}

/// Deterministic values in `[-1, 1)`, one per pseudo-random byte.
#[cfg(test)]
pub(crate) fn pseudo_random_values(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f32::from(state.to_be_bytes()[0]) / 128.0 - 1.0
        })
        .collect()
}
