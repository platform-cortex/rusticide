// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Where each tensor's bytes lie in the data section.

use std::collections::HashMap;

use half::f16;
use rusticide_core::Shape;
use rusticide_quantize::{BlockFormat, block_format};

use super::tensor_info::TensorInfo;
use crate::error::{GgufError, Result};

pub(super) const F32_TYPE_ID: u32 = 0;
const F16_TYPE_ID: u32 = 1;

/// The tensor's block format (`None` for F32 and F16) and its byte length.
pub(super) fn byte_len(info: &TensorInfo) -> Result<(Option<&'static BlockFormat>, usize)> {
    let numel = Shape::new(&info.shape).numel();
    let (format, len) = match info.type_id {
        F32_TYPE_ID => (None, numel.checked_mul(size_of::<f32>())),
        F16_TYPE_ID => (None, numel.checked_mul(size_of::<f16>())),
        type_id => {
            let format = block_format(type_id).ok_or_else(|| GgufError::UnsupportedTensorType {
                name: info.name.clone(),
                type_id: info.type_id,
            })?;
            (Some(format), format.byte_len(numel))
        }
    };
    let len = len.ok_or(GgufError::CastOverflow("tensor byte length"))?;
    Ok((format, len))
}

/// Rejects two tensors that share bytes: every loaded tensor is copied, so
/// aliased tensors would let a small file claim unbounded memory. A tensor
/// whose extent cannot be computed is skipped here and fails when loaded.
pub(super) fn ensure_disjoint(tensors: &HashMap<String, TensorInfo>) -> Result<()> {
    let mut extents: Vec<(usize, usize, &str)> = tensors
        .values()
        .filter_map(|info| {
            let (_, len) = byte_len(info).ok()?;
            let start = usize::try_from(info.offset).ok()?;
            let end = start.checked_add(len)?;
            (len > 0).then_some((start, end, info.name.as_str()))
        })
        .collect();
    extents.sort_unstable();
    let shared = extents.windows(2).find(|pair| pair[0].1 > pair[1].0);
    shared.map_or(Ok(()), |pair| {
        Err(GgufError::OverlappingTensors {
            first: pair[0].2.to_owned(),
            second: pair[1].2.to_owned(),
        })
    })
}

#[cfg(test)]
#[path = "extent_tests.rs"]
mod tests;
