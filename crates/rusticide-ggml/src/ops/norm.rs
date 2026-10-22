// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_core::{DType, Device, Error, Tensor};

use crate::Result;

/// Softmax over the last dimension. A scalar, or an input whose last
/// dimension is empty, has no row to normalise and yields zeros.
pub(super) fn softmax(input: &Tensor) -> Result<Tensor> {
    let mut values = input.to_vec::<f32>()?;
    let dims = input.shape().dims();
    let row_len = dims.last().copied().unwrap_or(0);
    if row_len == 0 {
        return Ok(Tensor::zeros(dims, DType::F32, Device::cpu())?);
    }
    for row in values.chunks_mut(row_len) {
        let max = row
            .iter()
            .fold(f32::NEG_INFINITY, |acc, &value| acc.max(value));
        let mut sum = 0.0;
        for value in row.iter_mut() {
            *value = (*value - max).exp();
            sum += *value;
        }
        for value in row.iter_mut() {
            *value /= sum;
        }
    }
    Ok(Tensor::from_slice(&values, dims)?)
}

/// Divides each last-dimension row by `sqrt(mean(x^2) + eps)`.
pub(super) fn rms_norm(input: &Tensor, eps: f32) -> Result<Tensor> {
    let mut values = input.to_vec::<f32>()?;
    let dims = input.shape().dims();
    let Some(&row_len) = dims.last() else {
        return Err(Error::RankMismatch {
            expected: 1,
            got: 0,
        }
        .into());
    };
    for row in values.chunks_mut(row_len.max(1)) {
        let mean_square =
            row.iter().map(|&value| value * value).sum::<f32>() / row_len_as_f32(row_len);
        let rms = (mean_square + eps).sqrt();
        for value in row.iter_mut() {
            *value /= rms;
        }
    }
    Ok(Tensor::from_slice(&values, dims)?)
}

#[expect(
    clippy::cast_precision_loss,
    reason = "row lengths beyond 2^24 lose low bits of the mean, an accepted f32 rounding"
)]
const fn row_len_as_f32(row_len: usize) -> f32 {
    row_len as f32
}

#[cfg(test)]
mod tests;
