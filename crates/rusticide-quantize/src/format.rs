// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The table of supported block formats, keyed by GGML type id.

use rusticide_core::{Shape, Tensor};

use crate::error::{QuantizeError, ensure_aligned, ensure_len, zeroed_f32};
use crate::{q4_0, q4_k, q5_0, q6_k, q8_0};

/// One block-quantized GGML format.
#[derive(Debug, Clone, Copy)]
pub struct BlockFormat {
    name: &'static str,
    type_id: u32,
    block_size: usize,
    block_bytes: usize,
    /// Receives exactly `block_bytes` bytes and `block_size` outputs.
    dequantize_block: fn(block: &[u8], out: &mut [f32]),
}

static FORMATS: [BlockFormat; 5] = [
    BlockFormat {
        name: "Q4_0",
        type_id: 2,
        block_size: q4_0::Q4_0_BLOCK_SIZE,
        block_bytes: q4_0::Q4_0_BLOCK_BYTES,
        dequantize_block: q4_0::dequantize_block,
    },
    BlockFormat {
        name: "Q5_0",
        type_id: 6,
        block_size: q5_0::Q5_0_BLOCK_SIZE,
        block_bytes: q5_0::Q5_0_BLOCK_BYTES,
        dequantize_block: q5_0::dequantize_block,
    },
    BlockFormat {
        name: "Q4_K",
        type_id: 12,
        block_size: q4_k::Q4_K_BLOCK_SIZE,
        block_bytes: q4_k::Q4_K_BLOCK_BYTES,
        dequantize_block: q4_k::dequantize_block,
    },
    BlockFormat {
        name: "Q6_K",
        type_id: 14,
        block_size: q6_k::Q6_K_BLOCK_SIZE,
        block_bytes: q6_k::Q6_K_BLOCK_BYTES,
        dequantize_block: q6_k::dequantize_block,
    },
    BlockFormat {
        name: "Q8_0",
        type_id: 8,
        block_size: q8_0::Q8_0_BLOCK_SIZE,
        block_bytes: q8_0::Q8_0_BLOCK_BYTES,
        dequantize_block: q8_0::dequantize_block,
    },
];

/// Every registered block format.
#[must_use]
pub fn formats() -> &'static [BlockFormat] {
    &FORMATS
}

/// The block format registered for a GGML type id, if any.
#[must_use]
pub fn block_format(type_id: u32) -> Option<&'static BlockFormat> {
    FORMATS.iter().find(|format| format.type_id == type_id)
}

impl BlockFormat {
    /// The GGML name of the format, such as `Q4_K`.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        self.name
    }

    /// The GGML type id.
    #[must_use]
    pub const fn type_id(&self) -> u32 {
        self.type_id
    }

    /// Elements per block.
    #[must_use]
    pub const fn block_size(&self) -> usize {
        self.block_size
    }

    /// Bytes per block.
    #[must_use]
    pub const fn block_bytes(&self) -> usize {
        self.block_bytes
    }

    /// Bytes holding `numel` elements, rounded up to whole blocks; `None` on
    /// overflow.
    #[must_use]
    pub const fn byte_len(&self, numel: usize) -> Option<usize> {
        numel
            .div_ceil(self.block_size)
            .checked_mul(self.block_bytes)
    }

    /// Dequantizes `data` to an F32 tensor of `shape`.
    ///
    /// # Errors
    ///
    /// Returns [`QuantizeError::UnalignedShape`] if the element count is not a
    /// multiple of the block size (checked first, so an overflowing shape
    /// lands here), or [`QuantizeError::DataTooShort`] if `data` is too short
    /// for the `shape`.
    pub fn dequantize(&self, data: &[u8], shape: &[usize]) -> Result<Tensor, QuantizeError> {
        let numel = Shape::new(shape).numel();
        ensure_aligned(self.name, numel, self.block_size)?;
        let needed = self.byte_len(numel).unwrap_or(usize::MAX);
        ensure_len(self.name, data, needed)?;

        let mut values = zeroed_f32(numel)?;
        for (block, out) in data
            .chunks(self.block_bytes)
            .zip(values.chunks_mut(self.block_size))
        {
            (self.dequantize_block)(block, out);
        }
        Ok(Tensor::from_slice(&values, shape)?)
    }
}

#[cfg(test)]
#[path = "format_tests.rs"]
mod tests;
