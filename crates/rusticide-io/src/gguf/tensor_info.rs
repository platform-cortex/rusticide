// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_core::{DType, Shape};
use rusticide_quantize::{
    Q4_0_BLOCK_BYTES, Q4_0_BLOCK_SIZE, Q4_K_BLOCK_BYTES, Q4_K_BLOCK_SIZE, Q6_K_BLOCK_BYTES,
    Q6_K_BLOCK_SIZE,
};

use super::cursor::Cursor;
use crate::error::{GgufError, Result};

/// GGML's maximum tensor rank.
const MAX_DIMENSIONS: u32 = 4;

/// The GGML storage types this reader understands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    /// 32-bit float.
    F32,
    /// 16-bit float.
    F16,
    /// 4-bit quantization, 32-element blocks.
    Q4_0,
    /// 4-bit k-quant.
    Q4K,
    /// 6-bit k-quant.
    Q6K,
}

impl GgmlType {
    /// Maps a GGML type tag, or `None` when unsupported.
    #[must_use]
    pub const fn from_raw(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            12 => Some(Self::Q4K),
            14 => Some(Self::Q6K),
            _ => None,
        }
    }

    /// The GGML type tag.
    #[must_use]
    pub const fn raw(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4K => 12,
            Self::Q6K => 14,
        }
    }

    /// The dtype a loaded tensor has: F16 for F16, F32 otherwise.
    #[must_use]
    pub const fn materialized_dtype(self) -> DType {
        match self {
            Self::F16 => DType::F16,
            Self::F32 | Self::Q4_0 | Self::Q4K | Self::Q6K => DType::F32,
        }
    }

    /// Bytes `numel` elements occupy on disk; `None` on overflow.
    #[must_use]
    pub const fn byte_len(self, numel: usize) -> Option<usize> {
        match self {
            Self::F32 => numel.checked_mul(4),
            Self::F16 => numel.checked_mul(2),
            Self::Q4_0 => block_bytes(numel, Q4_0_BLOCK_SIZE, Q4_0_BLOCK_BYTES),
            Self::Q4K => block_bytes(numel, Q4_K_BLOCK_SIZE, Q4_K_BLOCK_BYTES),
            Self::Q6K => block_bytes(numel, Q6_K_BLOCK_SIZE, Q6_K_BLOCK_BYTES),
        }
    }
}

const fn block_bytes(numel: usize, block_size: usize, block_bytes: usize) -> Option<usize> {
    numel.div_ceil(block_size).checked_mul(block_bytes)
}

/// One entry of the tensor-info table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    /// Tensor name.
    pub name: String,
    /// Extents, outermost first (GGUF stores them reversed); `[1]` for a scalar.
    pub shape: Vec<usize>,
    /// The dtype after loading.
    pub dtype: DType,
    /// Byte offset relative to the start of the tensor-data section.
    pub offset: u64,
    /// The on-disk storage type.
    pub ggml_type: GgmlType,
}

pub(crate) fn parse_tensor_info(cursor: &mut Cursor<'_>) -> Result<TensorInfo> {
    let name = cursor.string()?;
    let shape = parse_shape(cursor, &name)?;
    let tag = cursor.u32()?;
    let ggml_type = GgmlType::from_raw(tag).ok_or(GgufError::UnsupportedTensorType(tag))?;
    let offset = cursor.u64()?;
    Ok(TensorInfo {
        name,
        shape,
        dtype: ggml_type.materialized_dtype(),
        offset,
        ggml_type,
    })
}

fn parse_shape(cursor: &mut Cursor<'_>, name: &str) -> Result<Vec<usize>> {
    let dimension_count = cursor.u32()?;
    if dimension_count > MAX_DIMENSIONS {
        return Err(GgufError::TooManyDimensions {
            got: dimension_count,
            max: MAX_DIMENSIONS,
        });
    }
    let mut shape = Vec::new();
    for _ in 0..dimension_count {
        shape.push(cursor.len()?);
    }
    shape.reverse();
    if shape.is_empty() {
        shape.push(1);
    }
    if Shape::new(&shape).numel() == usize::MAX {
        return Err(GgufError::ShapeOverflow {
            name: name.to_owned(),
        });
    }
    Ok(shape)
}

#[cfg(test)]
#[path = "tensor_info_tests.rs"]
mod tests;
