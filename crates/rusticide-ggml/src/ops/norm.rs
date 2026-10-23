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
        rusticide_kernels::softmax(row);
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
        rusticide_kernels::rms_norm(row, eps);
    }
    Ok(Tensor::from_slice(&values, dims)?)
}

#[cfg(test)]
mod tests;
