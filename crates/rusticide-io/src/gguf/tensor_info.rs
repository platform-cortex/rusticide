// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_core::Shape;

use super::cursor::Cursor;
use crate::error::{GgufError, Result};

/// GGML's maximum tensor rank.
const MAX_DIMENSIONS: u32 = 4;

/// One entry of the tensor-info table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    /// Tensor name.
    pub name: String,
    /// Extents, outermost first (GGUF stores them reversed); `[1]` for a scalar.
    pub shape: Vec<usize>,
    /// Byte offset relative to the start of the tensor-data section.
    pub offset: u64,
    /// The raw GGML type id; whether it is loadable is decided by `tensor`.
    pub type_id: u32,
}

pub(crate) fn parse_tensor_info(cursor: &mut Cursor<'_>) -> Result<TensorInfo> {
    let name = cursor.string()?;
    let shape = parse_shape(cursor, &name)?;
    let type_id = cursor.u32()?;
    let offset = cursor.u64()?;
    Ok(TensorInfo {
        name,
        shape,
        offset,
        type_id,
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
