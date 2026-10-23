// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_core::{Error, Tensor};

use crate::Result;

/// Rotates adjacent column pairs of a `[rows, columns]` input, the angle of
/// pair `p` in row `r` being `r * theta^(-2p / columns)`. At most
/// `dim_pairs` pairs are rotated, and none that would run past the last
/// column.
pub(super) fn rope(input: &Tensor, theta: f32, dim_pairs: usize) -> Result<Tensor> {
    let mut values = input.to_vec::<f32>()?;
    let dims = input.shape().dims();
    let &[_, columns] = dims else {
        return Err(Error::RankMismatch {
            expected: 2,
            got: dims.len(),
        }
        .into());
    };
    for (position, row) in values.chunks_mut(columns.max(1)).enumerate() {
        rusticide_kernels::rope_adjacent(row, position, theta, dim_pairs);
    }
    Ok(Tensor::from_slice(&values, dims)?)
}

#[cfg(test)]
mod tests;
