// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use half::f16;
use rusticide_core::{DType, Tensor};
use rusticide_quantize::{BlockFormat, Q8Input};

use crate::error::{ModelError, Result};
use crate::source::ModelSource;

/// A row-major weight matrix of `rows` x `cols`. Block-quantized weights stay
/// quantized and are decoded a block at a time inside the dot product.
#[derive(Debug)]
pub struct Matrix {
    storage: Storage,
    rows: usize,
    cols: usize,
}

#[derive(Debug)]
enum Storage {
    Dense(Vec<f32>),
    Blocks {
        format: &'static BlockFormat,
        data: Vec<u8>,
        row_bytes: usize,
    },
}

impl Matrix {
    /// # Errors
    ///
    /// `MissingTensor` or `ShapeMismatch` when the source lacks the tensor or
    /// its shape is not `[rows, cols]`; a tensor error for unsupported dtypes.
    pub fn load(source: &dyn ModelSource, name: &str, rows: usize, cols: usize) -> Result<Self> {
        check_shape(source, name, &[rows, cols])?;
        let blocks = source
            .quantized_tensor(name)?
            .and_then(|(format, data)| Some((format, data, format.row_bytes(cols)?)));
        let storage = match blocks {
            Some((format, data, row_bytes)) => Storage::Blocks {
                format,
                data: data.to_vec(),
                row_bytes,
            },
            None => Storage::Dense(widen(&source.tensor(name)?)?),
        };
        Ok(Self {
            storage,
            rows,
            cols,
        })
    }

    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Whether the weights are held block-quantized.
    #[must_use]
    pub const fn is_quantized(&self) -> bool {
        matches!(self.storage, Storage::Blocks { .. })
    }

    /// Decodes row `index` into `out`; `false` when the row does not exist.
    pub fn copy_row(&self, index: usize, out: &mut [f32]) -> bool {
        let row = match &self.storage {
            Storage::Dense(data) => {
                let row = data.chunks_exact(self.cols).nth(index);
                row.map(|row| out.copy_from_slice(row))
            }
            Storage::Blocks {
                format,
                data,
                row_bytes,
            } => {
                let row = data.chunks_exact(*row_bytes).nth(index);
                row.map(|row| format.dequantize_row(row, out))
            }
        };
        row.is_some()
    }

    pub fn matvec(&self, input: &[f32], output: &mut [f32]) {
        match &self.storage {
            Storage::Dense(data) => rusticide_kernels::matvec(data, self.cols, input, output),
            Storage::Blocks {
                format,
                data,
                row_bytes,
            } => {
                let mut quantized = Q8Input::default();
                quantized.quantize(input, format.activation());
                rusticide_kernels::map_rows(data, *row_bytes, output, |row| {
                    format.dot_row_q8(row, &quantized)
                });
            }
        }
    }
}

/// # Errors
///
/// As [`Matrix::load`], for a tensor of shape `[len]`.
pub fn load_vector(source: &dyn ModelSource, name: &str, len: usize) -> Result<Vec<f32>> {
    load_checked(source, name, &[len])
}

fn load_checked(source: &dyn ModelSource, name: &str, expected: &[usize]) -> Result<Vec<f32>> {
    check_shape(source, name, expected)?;
    widen(&source.tensor(name)?)
}

fn check_shape(source: &dyn ModelSource, name: &str, expected: &[usize]) -> Result<()> {
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
    Ok(())
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
