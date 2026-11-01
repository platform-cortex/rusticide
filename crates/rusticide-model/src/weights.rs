// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use half::f16;
use rusticide_core::{DType, Tensor};
use rusticide_kernels::{Q8Kind, QuantKernel, active, dot};
use rusticide_quantize::BlockFormat;

use crate::batch::{Activations, matvec_batch};

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
        kernel: &'static QuantKernel,
        data: Vec<u8>,
        row_bytes: usize,
    },
}

impl Matrix {
    /// # Errors
    ///
    /// `MissingTensor` or `ShapeMismatch` when the source lacks the tensor or
    /// its shape is not `[rows, cols]`, `InvalidConfig` when `cols` is zero,
    /// `MissingKernel` when no kernel reads its block format, and a tensor
    /// error for unsupported dtypes.
    pub fn load(source: &dyn ModelSource, name: &str, rows: usize, cols: usize) -> Result<Self> {
        check_shape(source, name, &[rows, cols])?;
        if cols == 0 {
            return Err(ModelError::InvalidConfig {
                reason: "a weight matrix has no columns",
            });
        }
        let blocks = source
            .quantized_tensor(name)?
            .and_then(|(format, data)| Some((format, data, format.row_bytes(cols)?)));
        let storage = match blocks {
            Some((format, data, row_bytes)) => Storage::Blocks {
                format,
                kernel: require_kernel(format, active().quant_kernel(format.type_id()))?,
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
                ..
            } => {
                let row = data.chunks_exact(*row_bytes).nth(index);
                row.map(|row| format.dequantize_row(row, out))
            }
        };
        row.is_some()
    }

    pub fn matvec(&self, input: &[f32], output: &mut [f32]) {
        matvec_batch(input, &mut [(self, output)]);
    }

    /// The activation format the weights multiply against; `None` for
    /// dense weights, which take the input as is.
    pub(crate) const fn activation(&self) -> Option<Q8Kind> {
        match &self.storage {
            Storage::Dense(_) => None,
            Storage::Blocks { kernel, .. } => Some(kernel.activation()),
        }
    }

    /// Writes the products of rows `first_row..` with every token of the
    /// input into `out`, the tokens of one row side by side.
    pub(crate) fn compute_rows(&self, first_row: usize, out: &mut [f32], input: &Activations<'_>) {
        let tokens = input.tokens().max(1);
        match &self.storage {
            Storage::Dense(data) => {
                let rows = data.chunks_exact(self.cols).skip(first_row);
                for (slots, row) in out.chunks_mut(tokens).zip(rows) {
                    for (slot, token) in slots.iter_mut().zip(&input.dense) {
                        *slot = dot(row, token);
                    }
                }
            }
            Storage::Blocks {
                kernel,
                data,
                row_bytes,
                ..
            } => {
                let quantized = input.quantized(kernel.activation());
                let rows = data.chunks_exact(*row_bytes).skip(first_row);
                for (slots, row) in out.chunks_mut(tokens).zip(rows) {
                    kernel.dot_rows(row, quantized, slots);
                }
            }
        }
    }
}

fn require_kernel(
    format: &BlockFormat,
    kernel: Option<&'static QuantKernel>,
) -> Result<&'static QuantKernel> {
    kernel.ok_or_else(|| ModelError::MissingKernel {
        format: format.name(),
    })
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
