// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_core::{Error, Tensor, try_vec_with_capacity};

use crate::{GraphError, Result};

pub(super) fn get_rows(data: &Tensor, indices: &Tensor) -> Result<Tensor> {
    if indices.ndim() != 1 {
        return Err(rank_mismatch(indices.ndim()));
    }
    let Some((_, row_dims)) = data.shape().dims().split_first() else {
        return Err(rank_mismatch(0));
    };
    let values = data.to_vec::<f32>()?;
    let wanted = indices.to_vec::<f32>()?;
    let row_len: usize = row_dims.iter().product();
    let mut gathered = try_vec_with_capacity(wanted.len().saturating_mul(row_len))?;
    for &index in &wanted {
        let row = row_index(index, data.shape().dims())?;
        gathered.extend_from_slice(&values[row * row_len..(row + 1) * row_len]);
    }
    let mut dims = vec![wanted.len()];
    dims.extend_from_slice(row_dims);
    Ok(Tensor::from_slice(&gathered, &dims)?)
}

fn rank_mismatch(got: usize) -> GraphError {
    Error::RankMismatch { expected: 1, got }.into()
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is finite, non-negative and integral; a saturated oversized one fails the bounds check"
)]
fn row_index(index: f32, dims: &[usize]) -> Result<usize> {
    // NaN and the infinities have a NaN fract, so the fract test rejects them.
    if index < 0.0 || index.fract() != 0.0 {
        return Err(GraphError::InvalidIndex { value: index });
    }
    let row = index as usize;
    if row >= dims[0] {
        return Err(Error::IndexOutOfBounds {
            index: vec![row],
            shape: dims.to_vec(),
        }
        .into());
    }
    Ok(row)
}

#[cfg(test)]
mod tests;
