// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_core::Tensor;

pub(crate) const EPS: f32 = 1e-5;

pub(crate) fn tensor(shape: &[usize], data: &[f32]) -> Tensor {
    Tensor::from_slice(data, shape).unwrap()
}

pub(crate) fn zeros(shape: &[usize]) -> Tensor {
    tensor(shape, &vec![0.0; shape.iter().product()])
}

/// A `rows x cols` matrix holding `1.0` at each `(row, col)` and `0.0` elsewhere.
pub(crate) fn picker(rows: usize, cols: usize, picks: &[(usize, usize)]) -> Tensor {
    let mut data = vec![0.0; rows * cols];
    for &(row, col) in picks {
        data[row * cols + col] = 1.0;
    }
    tensor(&[rows, cols], &data)
}

pub(crate) fn identity(size: usize) -> Tensor {
    let picks: Vec<(usize, usize)> = (0..size).map(|index| (index, index)).collect();
    picker(size, size, &picks)
}

/// `row / sqrt(mean(row^2) + EPS)`, the reference rms norm without a weight.
pub(crate) fn normed(row: &[f32]) -> Vec<f32> {
    let count = f32::from(u8::try_from(row.len()).unwrap());
    let rms = (row.iter().map(|v| v * v).sum::<f32>() / count + EPS).sqrt();
    row.iter().map(|v| v / rms).collect()
}

/// Rotates adjacent pairs `(2p, 2p + 1)` by `position * base^(-2p / len)`.
pub(crate) fn rotated(row: &[f32], position: f32, base: f32) -> Vec<f32> {
    let len = f32::from(u8::try_from(row.len()).unwrap());
    let mut out = row.to_vec();
    for (pair, [low, high]) in out.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        let exponent = -2.0 * f32::from(u8::try_from(pair).unwrap()) / len;
        let (sin, cos) = (position * base.powf(exponent)).sin_cos();
        (*low, *high) = (
            low.mul_add(cos, -*high * sin),
            high.mul_add(cos, *low * sin),
        );
    }
    out
}

pub(crate) fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < 1e-5, "{actual:?} vs {expected:?}");
    }
}
