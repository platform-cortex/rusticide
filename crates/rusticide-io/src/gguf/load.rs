// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use half::f16;
use rusticide_core::{Element, Shape, Tensor, try_vec_with_capacity};
use rusticide_quantize::block_format;

use super::file::GgufFile;
use super::tensor_info::{GgmlType, TensorInfo};
use crate::error::{GgufError, Result};

impl GgufFile {
    /// Materializes a tensor by name.
    ///
    /// # Errors
    ///
    /// `TensorNotFound` for an unknown name, otherwise as loading the data.
    pub fn tensor(&self, name: &str) -> Result<Tensor> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_owned()))?;
        self.load_tensor(info)
    }

    /// Materializes the tensor at `index` in sorted-name order.
    ///
    /// # Errors
    ///
    /// `TensorNotFound` when out of range, otherwise as [`Self::tensor`].
    pub fn tensor_by_index(&self, index: usize) -> Result<Tensor> {
        let name = self
            .tensor_names
            .get(index)
            .ok_or_else(|| GgufError::TensorNotFound(format!("index {index}")))?;
        self.tensor(name)
    }

    fn load_tensor(&self, info: &TensorInfo) -> Result<Tensor> {
        let numel = Shape::new(&info.shape).numel();
        let len = info
            .ggml_type
            .byte_len(numel)
            .ok_or(GgufError::CastOverflow("tensor byte length"))?;
        let bytes = self.tensor_bytes(info, len)?;
        match info.ggml_type {
            GgmlType::F32 => decode_le(bytes, &info.shape, f32::from_le_bytes),
            GgmlType::F16 => decode_le(bytes, &info.shape, f16::from_le_bytes),
            GgmlType::Q4_0 | GgmlType::Q4K | GgmlType::Q6K => dequantize_block_format(info, bytes),
        }
    }

    fn tensor_bytes(&self, info: &TensorInfo, len: usize) -> Result<&[u8]> {
        let offset = usize::try_from(info.offset).unwrap_or(usize::MAX);
        let start = self
            .data_start
            .checked_add(offset)
            .ok_or(GgufError::UnexpectedEof)?;
        let end = start.checked_add(len).ok_or(GgufError::UnexpectedEof)?;
        self.data.get(start..end).ok_or(GgufError::UnexpectedEof)
    }
}

fn dequantize_block_format(info: &TensorInfo, bytes: &[u8]) -> Result<Tensor> {
    let raw = info.ggml_type.raw();
    let format = block_format(raw).ok_or(GgufError::UnsupportedTensorType(raw))?;
    Ok(format.dequantize(bytes, &info.shape)?)
}

fn decode_le<T: Element, const WIDTH: usize>(
    bytes: &[u8],
    shape: &[usize],
    convert: fn([u8; WIDTH]) -> T,
) -> Result<Tensor> {
    let (chunks, _) = bytes.as_chunks::<WIDTH>();
    let mut values = try_vec_with_capacity(chunks.len())?;
    values.extend(chunks.iter().map(|chunk| convert(*chunk)));
    Ok(Tensor::from_slice(&values, shape)?)
}

#[cfg(test)]
#[path = "load_tests.rs"]
mod tests;
