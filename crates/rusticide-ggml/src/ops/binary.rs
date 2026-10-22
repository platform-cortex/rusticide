// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_core::{EagerOps, Shape, Tensor, try_vec_with_capacity};

use crate::Result;
use crate::index::{broadcast_map, flatten_index, unflatten_index};

pub(super) fn add(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    Ok(lhs.add(rhs)?)
}

pub(super) fn matmul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    Ok(lhs.matmul(rhs)?)
}

pub(super) fn mul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    let lhs_values = lhs.to_vec::<f32>()?;
    let rhs_values = rhs.to_vec::<f32>()?;
    let (lhs_dims, rhs_dims) = (lhs.shape().dims(), rhs.shape().dims());
    let shape = Shape::new(lhs_dims).broadcast_with(&Shape::new(rhs_dims))?;
    let dims = shape.dims();
    let mut values = try_vec_with_capacity(shape.numel())?;
    for flat in 0..shape.numel() {
        let at = unflatten_index(flat, dims);
        let lhs_at = flatten_index(&broadcast_map(&at, lhs_dims), lhs_dims);
        let rhs_at = flatten_index(&broadcast_map(&at, rhs_dims), rhs_dims);
        values.push(lhs_values[lhs_at] * rhs_values[rhs_at]);
    }
    Ok(Tensor::from_slice(&values, dims)?)
}
