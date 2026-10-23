// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use half::f16;
use rusticide_core::{DType, Tensor};

use crate::error::{ModelError, Result};
use crate::source::ModelSource;

/// A row-major `f32` weight matrix of `rows` x `cols`.
#[derive(Debug)]
pub struct Matrix {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    /// # Errors
    ///
    /// `MissingTensor` or `ShapeMismatch` when the source lacks the tensor or
    /// its shape is not `[rows, cols]`; a tensor error for unsupported dtypes.
    pub fn load(source: &dyn ModelSource, name: &str, rows: usize, cols: usize) -> Result<Self> {
        let data = load_checked(source, name, &[rows, cols])?;
        Ok(Self { data, rows, cols })
    }

    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    #[must_use]
    pub fn row(&self, index: usize) -> Option<&[f32]> {
        let start = index.checked_mul(self.cols)?;
        self.data.get(start..start.checked_add(self.cols)?)
    }

    pub fn matvec(&self, input: &[f32], output: &mut [f32]) {
        rusticide_kernels::matvec(&self.data, self.cols, input, output);
    }
}

/// # Errors
///
/// As [`Matrix::load`], for a tensor of shape `[len]`.
pub fn load_vector(source: &dyn ModelSource, name: &str, len: usize) -> Result<Vec<f32>> {
    load_checked(source, name, &[len])
}

fn load_checked(source: &dyn ModelSource, name: &str, expected: &[usize]) -> Result<Vec<f32>> {
    let got = source
        .tensor_shape(name)
        .ok_or_else(|| ModelError::MissingTensor {
            name: name.to_owned(),
        })?;
    if got != expected {
        return Err(ModelError::ShapeMismatch {
            name: name.to_owned(),
            expected: expected.to_vec(),
            got,
        });
    }
    widen(&source.tensor(name)?)
}

fn widen(tensor: &Tensor) -> Result<Vec<f32>> {
    if tensor.dtype() == DType::F16 {
        let halves = tensor.to_vec::<f16>()?;
        return Ok(halves.into_iter().map(f32::from).collect());
    }
    Ok(tensor.to_vec::<f32>()?)
}

#[cfg(test)]
#[path = "weights_tests.rs"]
mod tests;
