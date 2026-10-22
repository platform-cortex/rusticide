// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_core::{EagerOps, Shape, Tensor, try_vec_with_capacity};

use crate::Result;
use crate::index::{broadcast_map, ensure_repeatable, flatten_index, unflatten_index};

pub(super) fn copy(input: &Tensor) -> Result<Tensor> {
    Ok(input.deep_copy()?)
}

pub(super) fn reshape(input: &Tensor, shape: &[usize]) -> Result<Tensor> {
    Ok(input.reshape(shape)?)
}

pub(super) fn repeat(input: &Tensor, target: &[usize]) -> Result<Tensor> {
    let source = input.to_vec::<f32>()?;
    let source_dims = input.shape().dims();
    ensure_repeatable(source_dims, target)?;
    let count = Shape::new(target).numel();
    let mut values = try_vec_with_capacity::<f32>(count)?;
    for flat in 0..count {
        let at = broadcast_map(&unflatten_index(flat, target), source_dims);
        values.push(source[flatten_index(&at, source_dims)]);
    }
    Ok(Tensor::from_slice(&values, target)?)
}
