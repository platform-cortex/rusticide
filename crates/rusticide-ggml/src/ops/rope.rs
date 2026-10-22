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
        for pair in 0..dim_pairs.min(columns / 2) {
            let frequency = theta.powf(-lossy(2 * pair) / lossy(columns));
            let (sin, cos) = (lossy(position) * frequency).sin_cos();
            let (low, high) = (row[2 * pair], row[2 * pair + 1]);
            row[2 * pair] = low.mul_add(cos, -high * sin);
            row[2 * pair + 1] = high.mul_add(cos, low * sin);
        }
    }
    Ok(Tensor::from_slice(&values, dims)?)
}

#[expect(
    clippy::cast_precision_loss,
    reason = "positions and widths beyond 2^24 lose low bits of the angle, an accepted f32 rounding"
)]
const fn lossy(count: usize) -> f32 {
    count as f32
}

#[cfg(test)]
mod tests;
