// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::adapter::{Adapter, active};

fn adapter() -> &'static Adapter {
    active()
}

fn route_dot(adapter: &Adapter, lhs: &[f32], rhs: &[f32]) -> f32 {
    (adapter.f32.dot)(lhs, rhs)
}

/// Dot product. The shorter slice bounds the work.
#[must_use]
pub fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    route_dot(adapter(), lhs, rhs)
}

/// Scores every `row_stride`-wide row of `rows`: `out[i]` is
/// `dot(query, rows_i[offset..]) / scale`. The shortest of `out`, the rows and
/// `query` bounds the work; a zero stride is treated as one.
pub fn dot_rows(
    query: &[f32],
    rows: &[f32],
    row_stride: usize,
    offset: usize,
    scale: f32,
    out: &mut [f32],
) {
    (adapter().f32.dot_rows)(query, rows, row_stride, offset, scale, out);
}

/// Element-wise `target += addend`; the shorter slice bounds the work.
pub fn add_assign(target: &mut [f32], addend: &[f32]) {
    (adapter().f32.add_assign)(target, addend);
}

/// Element-wise `target *= factor`; the shorter slice bounds the work.
pub fn mul_assign(target: &mut [f32], factor: &[f32]) {
    (adapter().f32.mul_assign)(target, factor);
}

/// In-place sigmoid linear unit: `value / (1 + exp(-value))` per element.
pub fn silu_assign(values: &mut [f32]) {
    (adapter().f32.silu_assign)(values);
}

/// Divides the row by `sqrt(mean(x^2) + eps)`. An empty row is a no-op.
pub fn rms_norm(row: &mut [f32], eps: f32) {
    (adapter().f32.rms_norm)(row, eps);
}

/// Max-subtracted softmax in place. An empty row is a no-op.
pub fn softmax(row: &mut [f32]) {
    (adapter().f32.softmax)(row);
}

/// Rotates adjacent pairs `(2p, 2p+1)` for `p < min(dim_pairs, row.len() / 2)`
/// by angle `position * theta^(-2p / row.len())`.
pub fn rope_adjacent(row: &mut [f32], position: usize, theta: f32, dim_pairs: usize) {
    (adapter().f32.rope_adjacent)(row, position, theta, dim_pairs);
}

/// Rotates the half-split pairs `(p, p + row.len() / 2)` for
/// `p < min(dim_pairs, row.len() / 2)` by angle
/// `position * theta^(-2p / row.len())`.
pub fn rope_half_split(row: &mut [f32], position: usize, theta: f32, dim_pairs: usize) {
    (adapter().f32.rope_half_split)(row, position, theta, dim_pairs);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::portable;

    #[test]
    fn free_functions_route_through_the_adapter() {
        let mut kernels = portable::ADAPTER.f32;
        kernels.dot = |_, _| 42.0;
        let custom = Adapter {
            name: "custom",
            f32: kernels,
        };
        assert_eq!(route_dot(&custom, &[1.0], &[1.0]), 42.0);
        assert_eq!(dot(&[1.0, 2.0], &[3.0, 4.0]), 11.0);
    }
}
