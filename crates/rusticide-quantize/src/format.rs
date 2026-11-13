// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The table of supported block formats, keyed by GGML type id.

use rusticide_core::{Shape, Tensor};

use crate::error::{QuantizeError, ensure_aligned, ensure_len, zeroed_f32};
use crate::{i2_s, q4_0, q4_k, q5_0, q6_k, q8_0};

/// Applies a tensor's trailer to all of its dequantized values.
type ApplyTrailer = fn(trailer: &[u8], values: &mut [f32]);

/// One block-quantized GGML format.
#[derive(Debug, Clone, Copy)]
pub struct BlockFormat {
    name: &'static str,
    type_id: u32,
    block_size: usize,
    block_bytes: usize,
    /// Receives exactly `block_bytes` bytes and `block_size` outputs.
    dequantize_block: fn(block: &[u8], out: &mut [f32]),
    /// Byte offsets of the f16 scale fields within a block.
    scale_offsets: &'static [usize],
    /// Bytes after the last block holding data for the whole tensor.
    trailer_bytes: usize,
    apply_trailer: Option<ApplyTrailer>,
}

static FORMATS: [BlockFormat; 6] = [
    BlockFormat {
        name: "Q4_0",
        type_id: 2,
        block_size: q4_0::Q4_0_BLOCK_SIZE,
        block_bytes: q4_0::Q4_0_BLOCK_BYTES,
        dequantize_block: q4_0::dequantize_block,
        scale_offsets: &[0],
        trailer_bytes: 0,
        apply_trailer: None,
    },
    BlockFormat {
        name: "Q5_0",
        type_id: 6,
        block_size: q5_0::Q5_0_BLOCK_SIZE,
        block_bytes: q5_0::Q5_0_BLOCK_BYTES,
        dequantize_block: q5_0::dequantize_block,
        scale_offsets: &[0],
        trailer_bytes: 0,
        apply_trailer: None,
    },
    BlockFormat {
        name: "Q4_K",
        type_id: 12,
        block_size: q4_k::Q4_K_BLOCK_SIZE,
        block_bytes: q4_k::Q4_K_BLOCK_BYTES,
        dequantize_block: q4_k::dequantize_block,
        scale_offsets: &[0, 2],
        trailer_bytes: 0,
        apply_trailer: None,
    },
    BlockFormat {
        name: "Q6_K",
        type_id: 14,
        block_size: q6_k::Q6_K_BLOCK_SIZE,
        block_bytes: q6_k::Q6_K_BLOCK_BYTES,
        dequantize_block: q6_k::dequantize_block,
        scale_offsets: &[208],
        trailer_bytes: 0,
        apply_trailer: None,
    },
    BlockFormat {
        name: "Q8_0",
        type_id: 8,
        block_size: q8_0::Q8_0_BLOCK_SIZE,
        block_bytes: q8_0::Q8_0_BLOCK_BYTES,
        dequantize_block: q8_0::dequantize_block,
        scale_offsets: &[0],
        trailer_bytes: 0,
        apply_trailer: None,
    },
    BlockFormat {
        name: "I2_S",
        type_id: 36,
        block_size: i2_s::I2_S_BLOCK_SIZE,
        block_bytes: i2_s::I2_S_BLOCK_BYTES,
        dequantize_block: i2_s::dequantize_block,
        scale_offsets: &[],
        trailer_bytes: i2_s::I2_S_TRAILER_BYTES,
        apply_trailer: Some(i2_s::apply_trailer),
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

const SYNTHETIC_SCALES: [f32; 7] = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07];

/// Deterministic bytes over the whole `u8` range.
fn pseudo_random_bytes(len: usize, seed: u32) -> Vec<u8> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state.to_be_bytes()[0]
        })
        .collect()
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

    /// Bytes after the last block holding data for the whole tensor, such
    /// as the `I2_S` scale; zero for formats scaled per block.
    #[must_use]
    pub const fn trailer_bytes(&self) -> usize {
        self.trailer_bytes
    }

    /// Bytes holding a tensor of `numel` elements: whole blocks, rounded up,
    /// then the trailer; `None` on overflow.
    #[must_use]
    pub const fn byte_len(&self, numel: usize) -> Option<usize> {
        match self.blocks_len(numel) {
            Some(blocks) => blocks.checked_add(self.trailer_bytes),
            None => None,
        }
    }

    const fn blocks_len(&self, numel: usize) -> Option<usize> {
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
        self.dequantize_row(data, &mut values);
        if let Some(apply) = self.apply_trailer {
            let trailer = needed.saturating_sub(self.trailer_bytes)..needed;
            apply(data.get(trailer).unwrap_or_default(), &mut values);
        }
        Ok(Tensor::from_slice(&values, shape)?)
    }

    /// Bytes of one row of `cols` elements, without the tensor's trailer;
    /// `None` unless `cols` is a whole number of blocks, or on overflow.
    #[must_use]
    pub const fn row_bytes(&self, cols: usize) -> Option<usize> {
        if !cols.is_multiple_of(self.block_size) {
            return None;
        }
        self.blocks_len(cols)
    }

    /// `blocks` blocks of pseudo-random bytes over the whole `u8` range, with
    /// every f16 scale field set to a finite value so no block dequantizes to
    /// infinity or NaN; `None` when the byte length overflows.
    #[must_use]
    pub fn synthetic_blocks(&self, blocks: usize, seed: u32) -> Option<Vec<u8>> {
        let mut data = pseudo_random_bytes(blocks.checked_mul(self.block_bytes)?, seed);
        for (index, block) in data.chunks_exact_mut(self.block_bytes).enumerate() {
            let scale = SYNTHETIC_SCALES[index % SYNTHETIC_SCALES.len()];
            for &offset in self.scale_offsets {
                block[offset..offset + 2]
                    .copy_from_slice(&half::f16::from_f32(scale).to_le_bytes());
            }
        }
        Some(data)
    }

    /// Dequantizes one row of whole blocks into `out`. The trailer is not
    /// applied: `I2_S` rows hold the raw `code - 1` values, and the caller
    /// multiplies by [`crate::i2_s_scale`].
    pub fn dequantize_row(&self, row: &[u8], out: &mut [f32]) {
        for (block, values) in row
            .chunks_exact(self.block_bytes)
            .zip(out.chunks_exact_mut(self.block_size))
        {
            (self.dequantize_block)(block, values);
        }
    }
}

#[cfg(test)]
#[path = "format_tests.rs"]
mod tests;
