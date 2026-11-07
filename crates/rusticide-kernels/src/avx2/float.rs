// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The portable `f32` kernels, with the fused dot products compiled for
//! `fma` once detection stored the proof.

use rusticide_simd::{avx2, float};

use super::{Proof, proof};
use crate::adapter::F32Kernels;
use crate::portable;

pub(super) static KERNELS: F32Kernels = F32Kernels {
    dot,
    dot_rows,
    mix_rows,
    ..portable::ADAPTER.f32
};

fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    dot_with(proof(), lhs, rhs)
}

fn dot_rows(
    query: &[f32],
    rows: &[f32],
    row_stride: usize,
    offset: usize,
    scale: f32,
    out: &mut [f32],
) {
    dot_rows_with(proof(), query, rows, row_stride, offset, scale, out);
}

fn mix_rows(weights: &[f32], rows: &[f32], row_stride: usize, offset: usize, out: &mut [f32]) {
    mix_rows_with(proof(), weights, rows, row_stride, offset, out);
}

fn dot_with(proof: Option<Proof>, lhs: &[f32], rhs: &[f32]) -> f32 {
    proof.map_or_else(
        || float::dot(lhs, rhs),
        |token| avx2::float::dot(token, lhs, rhs),
    )
}

fn dot_rows_with(
    proof: Option<Proof>,
    query: &[f32],
    rows: &[f32],
    row_stride: usize,
    offset: usize,
    scale: f32,
    out: &mut [f32],
) {
    match proof {
        Some(token) => {
            avx2::float::dot_rows(token, query, rows, row_stride, offset, scale, out);
        }
        None => float::dot_rows(query, rows, row_stride, offset, scale, out),
    }
}

fn mix_rows_with(
    proof: Option<Proof>,
    weights: &[f32],
    rows: &[f32],
    row_stride: usize,
    offset: usize,
    out: &mut [f32],
) {
    match proof {
        Some(token) => avx2::float::mix_rows(token, weights, rows, row_stride, offset, out),
        None => float::mix_rows(weights, rows, row_stride, offset, out),
    }
}

#[cfg(test)]
mod tests {
    use rusticide_simd::{Avx2, float};

    use super::super::ADAPTER;
    use super::{KERNELS, dot_rows_with, dot_with, mix_rows_with};

    fn values(seed: u64, len: usize) -> Vec<f32> {
        let mut rng = fastrand::Rng::with_seed(seed);
        (0..len)
            .map(|_| rng.f32().mul_add(2.0, -1.0) * 1e3)
            .collect()
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    #[test]
    fn the_adapter_runs_these_kernels() {
        assert!(std::ptr::fn_addr_eq(ADAPTER.f32.dot, KERNELS.dot));
        assert!(std::ptr::fn_addr_eq(ADAPTER.f32.dot_rows, KERNELS.dot_rows));
        assert!(std::ptr::fn_addr_eq(ADAPTER.f32.mix_rows, KERNELS.mix_rows));
    }

    #[test]
    fn with_and_without_a_token_they_match_portable_bit_for_bit() {
        let (lhs, rows) = (values(1, 40), values(2, 6 * 48));
        for proof in [Avx2::detect(), None] {
            let want = float::dot(&lhs, &rows);
            assert_eq!(dot_with(proof, &lhs, &rows).to_bits(), want.to_bits());

            let (mut got, mut want) = ([0.0; 6], [0.0; 6]);
            dot_rows_with(proof, &lhs, &rows, 48, 5, 3.0, &mut got);
            float::dot_rows(&lhs, &rows, 48, 5, 3.0, &mut want);
            assert_eq!(bits(&got), bits(&want));

            let (mut got, mut want) = (values(3, 40), values(3, 40));
            mix_rows_with(proof, &lhs[..6], &rows, 48, 7, &mut got);
            float::mix_rows(&lhs[..6], &rows, 48, 7, &mut want);
            assert_eq!(bits(&got), bits(&want));
        }
    }
}
