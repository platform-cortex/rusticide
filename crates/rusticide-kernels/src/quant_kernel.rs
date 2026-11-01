// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::q8_input::{Q8Input, Q8Kind};

pub(crate) type DotRow = fn(row: &[u8], input: &Q8Input) -> f32;
pub(crate) type DotRows = fn(row: &[u8], inputs: &[Q8Input], out: &mut [f32]);

/// The row kernels for one block format, keyed by its GGML type id, and the
/// activation scheme they pair with.
#[derive(Debug, Clone, Copy)]
pub struct QuantKernel {
    pub(crate) type_id: u32,
    pub(crate) activation: Q8Kind,
    pub(crate) dot: DotRow,
    /// One row against two or more inputs, when that beats one `dot` each.
    pub(crate) dot_rows: Option<DotRows>,
}

impl QuantKernel {
    /// The GGML type id of the block format these kernels read.
    #[must_use]
    pub const fn type_id(&self) -> u32 {
        self.type_id
    }

    /// How activations multiplied against this format are quantized.
    #[must_use]
    pub const fn activation(&self) -> Q8Kind {
        self.activation
    }

    /// Dot product of one quantized row with a quantized activation, in
    /// integer arithmetic. The shorter of the two bounds the work.
    #[must_use]
    pub fn dot_row(&self, row: &[u8], input: &Q8Input) -> f32 {
        (self.dot)(row, input)
    }

    /// [`Self::dot_row`] of one row with each input, into the matching slot
    /// of `out`; every value is bit-identical to its own `dot_row`.
    pub fn dot_rows(&self, row: &[u8], inputs: &[Q8Input], out: &mut [f32]) {
        match (self.dot_rows, inputs) {
            (Some(dot_rows), [_, _, ..]) => dot_rows(row, inputs, out),
            _ => {
                for (slot, input) in out.iter_mut().zip(inputs) {
                    *slot = (self.dot)(row, input);
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "quant_kernel_tests.rs"]
mod tests;
