// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The table of supported block formats, keyed by GGML type id.

use rusticide_core::{Shape, Tensor};

use crate::error::{QuantizeError, ensure_aligned, ensure_len, zeroed_f32};
use crate::q8_input::{Q8Input, Q8Kind};
use crate::{int_dot, q4_0, q4_k, q5_0, q6_k, q8_0};

/// One row against several inputs, one output slot per input.
type DotRows = fn(row: &[u8], inputs: &[Q8Input], out: &mut [f32]);

/// One block-quantized GGML format.
#[derive(Debug, Clone, Copy)]
pub struct BlockFormat {
    name: &'static str,
    type_id: u32,
    block_size: usize,
    block_bytes: usize,
    /// Receives exactly `block_bytes` bytes and `block_size` outputs.
    dequantize_block: fn(block: &[u8], out: &mut [f32]),
    /// One row of whole blocks against a [`Q8Input`].
    dot_q8: fn(row: &[u8], input: &Q8Input) -> f32,
    /// One row against two or more inputs, when that beats one `dot_q8` each.
    dot_rows: Option<DotRows>,
    activation: Q8Kind,
}

static FORMATS: [BlockFormat; 5] = [
    BlockFormat {
        name: "Q4_0",
        type_id: 2,
        block_size: q4_0::Q4_0_BLOCK_SIZE,
        block_bytes: q4_0::Q4_0_BLOCK_BYTES,
        dequantize_block: q4_0::dequantize_block,
        dot_q8: int_dot::q4_0,
        dot_rows: None,
        activation: Q8Kind::Block,
    },
    BlockFormat {
        name: "Q5_0",
        type_id: 6,
        block_size: q5_0::Q5_0_BLOCK_SIZE,
        block_bytes: q5_0::Q5_0_BLOCK_BYTES,
        dequantize_block: q5_0::dequantize_block,
        dot_q8: int_dot::q5_0,
        dot_rows: Some(int_dot::q5_0_rows),
        activation: Q8Kind::Block,
    },
    BlockFormat {
        name: "Q4_K",
        type_id: 12,
        block_size: q4_k::Q4_K_BLOCK_SIZE,
        block_bytes: q4_k::Q4_K_BLOCK_BYTES,
        dequantize_block: q4_k::dequantize_block,
        dot_q8: int_dot::q4_k,
        dot_rows: Some(int_dot::q4_k_rows),
        activation: Q8Kind::SuperBlock,
    },
    BlockFormat {
        name: "Q6_K",
        type_id: 14,
        block_size: q6_k::Q6_K_BLOCK_SIZE,
        block_bytes: q6_k::Q6_K_BLOCK_BYTES,
        dequantize_block: q6_k::dequantize_block,
        dot_q8: int_dot::q6_k,
        dot_rows: Some(int_dot::q6_k_rows),
        activation: Q8Kind::SuperBlock,
    },
    BlockFormat {
        name: "Q8_0",
        type_id: 8,
        block_size: q8_0::Q8_0_BLOCK_SIZE,
        block_bytes: q8_0::Q8_0_BLOCK_BYTES,
        dequantize_block: q8_0::dequantize_block,
        dot_q8: int_dot::q8_0,
        dot_rows: None,
        activation: Q8Kind::Block,
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

    /// Bytes of one row of `cols` elements; `None` unless `cols` is a whole
    /// number of blocks, or on overflow.
    #[must_use]
    pub const fn row_bytes(&self, cols: usize) -> Option<usize> {
        if !cols.is_multiple_of(self.block_size) {
            return None;
        }
        self.byte_len(cols)
    }

    /// How activations multiplied against this format are quantized.
    #[must_use]
    pub const fn activation(&self) -> Q8Kind {
        self.activation
    }

    /// Dot product of one quantized row with a quantized activation, in
    /// integer arithmetic. The shorter of the two bounds the work.
    #[must_use]
    pub fn dot_row_q8(&self, row: &[u8], input: &Q8Input) -> f32 {
        (self.dot_q8)(row, input)
    }

    /// [`Self::dot_row_q8`] of one row with each input, into the matching
    /// slot of `out`; every value is bit-identical to its own `dot_row_q8`.
    pub fn dot_rows_q8(&self, row: &[u8], inputs: &[Q8Input], out: &mut [f32]) {
        match (self.dot_rows, inputs) {
            (Some(dot_rows), [_, _, ..]) => dot_rows(row, inputs, out),
            _ => {
                for (slot, input) in out.iter_mut().zip(inputs) {
                    *slot = (self.dot_q8)(row, input);
                }
            }
        }
    }

    /// Dequantizes one row of whole blocks into `out`.
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
