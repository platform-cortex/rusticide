// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_core::{DType, Error, Shape};

use crate::index::ensure_repeatable;
use crate::{Op, Result};

type Input<'a> = (&'a [usize], DType);

/// The output shape and dtype of `op` applied to inputs of the given shapes
/// and dtypes, or the error executing it would raise.
pub(crate) fn infer_shape_dtype(op: &Op, inputs: &[Input<'_>]) -> Result<(Vec<usize>, DType)> {
    match op {
        Op::Copy | Op::Cont | Op::Permute => Ok((inputs[0].0.to_vec(), inputs[0].1)),
        Op::Add => infer_add(inputs[0], inputs[1]),
        Op::MatMul => infer_matmul(inputs[0].0, inputs[1].0),
        Op::Reshape { shape } => infer_reshape(inputs[0], shape),
        Op::Repeat { shape } => {
            ensure_repeatable(inputs[0].0, shape)?;
            Ok((shape.clone(), inputs[0].1))
        }
        Op::GetRows => infer_get_rows(inputs[0], inputs[1].0),
    }
}

fn infer_add(lhs: Input<'_>, rhs: Input<'_>) -> Result<(Vec<usize>, DType)> {
    if lhs.1 != rhs.1 {
        return Err(Error::DTypeMismatch {
            expected: lhs.1,
            got: rhs.1,
        }
        .into());
    }
    let shape = Shape::new(lhs.0).broadcast_with(&Shape::new(rhs.0))?;
    Ok((shape.dims().to_vec(), lhs.1))
}

fn infer_matmul(lhs: &[usize], rhs: &[usize]) -> Result<(Vec<usize>, DType)> {
    for dims in [lhs, rhs] {
        if dims.len() != 2 {
            return Err(Error::RankMismatch {
                expected: 2,
                got: dims.len(),
            }
            .into());
        }
    }
    if lhs[1] != rhs[0] {
        return Err(Error::ShapeMismatch {
            expected: lhs.to_vec(),
            got: rhs.to_vec(),
        }
        .into());
    }
    Ok((vec![lhs[0], rhs[1]], DType::F32))
}

fn infer_reshape(input: Input<'_>, target: &[usize]) -> Result<(Vec<usize>, DType)> {
    let expected = Shape::new(input.0).numel();
    let got = Shape::new(target).numel();
    if expected != got {
        return Err(Error::ElementCountMismatch { expected, got }.into());
    }
    Ok((target.to_vec(), input.1))
}

fn infer_get_rows(data: Input<'_>, indices: &[usize]) -> Result<(Vec<usize>, DType)> {
    if indices.len() != 1 {
        return Err(Error::RankMismatch {
            expected: 1,
            got: indices.len(),
        }
        .into());
    }
    let Some((_, row_dims)) = data.0.split_first() else {
        return Err(Error::RankMismatch {
            expected: 1,
            got: 0,
        }
        .into());
    };
    let mut shape = indices.to_vec();
    shape.extend_from_slice(row_dims);
    Ok((shape, data.1))
}

#[cfg(test)]
mod tests;
