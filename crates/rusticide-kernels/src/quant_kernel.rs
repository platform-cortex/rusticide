// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::q8_input::{Q8Input, Q8Kind};

pub(crate) type DotRow = fn(row: &[u8], input: &Q8Input) -> f32;
pub(crate) type DotRows = fn(row: &[u8], inputs: &[Q8Input], out: &mut [f32]);

/// A weight layout that stores `group_rows` rows together, always more
/// than one; a kernel without one reads row-major data.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Layout {
    pub(crate) group_rows: usize,
    /// Repacks a row-major matrix into this layout in place; `false`, with
    /// the data untouched, when it does not split into whole groups.
    pub(crate) prepare: fn(data: &mut [u8], row_bytes: usize) -> bool,
    /// Writes row `index`'s original bytes from the prepared layout.
    pub(crate) read_row: fn(data: &[u8], row_bytes: usize, index: usize, out: &mut [u8]) -> bool,
    /// One group against every input, into `out[row * inputs.len() + token]`.
    pub(crate) dot_group: fn(group: &[u8], inputs: &[Q8Input], out: &mut [f32]),
}

/// The row kernels for one block format, keyed by its GGML type id, the
/// activation scheme they pair with, and the weight layout they read.
#[derive(Debug, Clone, Copy)]
pub struct QuantKernel {
    pub(crate) type_id: u32,
    pub(crate) activation: Q8Kind,
    pub(crate) dot: DotRow,
    /// One row against two or more inputs, when that beats one `dot` each.
    pub(crate) dot_rows: Option<DotRows>,
    /// The grouped layout this kernel reads; `None` for row-major data.
    pub(crate) layout: Option<Layout>,
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

    /// Writes the original block bytes of row `index` of a matrix in this
    /// kernel's layout into `out`; `false` when the row does not exist or
    /// `out` is not `row_bytes` long.
    pub fn read_row(&self, data: &[u8], row_bytes: usize, index: usize, out: &mut [u8]) -> bool {
        let read = self
            .layout
            .map_or(read_row_major as _, |layout| layout.read_row);
        read(data, row_bytes, index, out)
    }

    /// Calls `visit` with the original block bytes of row `index`, borrowed
    /// from row-major data and copied out of a grouped layout; `false`, with
    /// no call, when the row does not exist.
    pub fn visit_row(
        &self,
        data: &[u8],
        row_bytes: usize,
        index: usize,
        visit: impl FnOnce(&[u8]),
    ) -> bool {
        if self.layout.is_none() {
            return row_major(data, row_bytes, index).map(visit).is_some();
        }
        let mut row = vec![0; row_bytes];
        let found = self.read_row(data, row_bytes, index, &mut row);
        if found {
            visit(&row);
        }
        found
    }

    /// The products of every row of `data`, in this kernel's layout, with
    /// each input, on the calling thread: `out[row * inputs.len() + token]`.
    pub fn dot_matrix(&self, data: &[u8], row_bytes: usize, inputs: &[Q8Input], out: &mut [f32]) {
        self.compute_rows(data, row_bytes, 0, inputs, out);
    }

    /// Rows a scheduler task must keep together.
    pub(crate) fn group_rows(&self) -> usize {
        self.layout.map_or(1, |layout| layout.group_rows)
    }

    /// The products of the rows from `first_row` (a group boundary) with
    /// every input, the inputs of one row side by side in `out`.
    pub(crate) fn compute_rows(
        &self,
        data: &[u8],
        row_bytes: usize,
        first_row: usize,
        inputs: &[Q8Input],
        out: &mut [f32],
    ) {
        let tokens = inputs.len().max(1);
        if let Some(layout) = self.layout {
            let group = layout.group_rows;
            // Only a matrix `prepare` accepted reaches a grouped kernel, and
            // it accepts whole groups only; `chunks_exact` would skip a
            // trailing partial group rather than read past the data.
            let groups = data
                .chunks_exact(row_bytes.saturating_mul(group))
                .skip(first_row / group);
            for (slots, rows) in out
                .chunks_exact_mut(group.saturating_mul(tokens))
                .zip(groups)
            {
                (layout.dot_group)(rows, inputs, slots);
            }
        } else {
            let rows = data.chunks_exact(row_bytes).skip(first_row);
            for (slots, row) in out.chunks_mut(tokens).zip(rows) {
                self.dot_rows(row, inputs, slots);
            }
        }
    }
}

/// [`QuantKernel::read_row`] for a row-major layout.
pub(crate) fn read_row_major(data: &[u8], row_bytes: usize, index: usize, out: &mut [u8]) -> bool {
    match row_major(data, row_bytes, index) {
        Some(row) if row.len() == out.len() => {
            out.copy_from_slice(row);
            true
        }
        _ => false,
    }
}

/// Row `index` of row-major data; `None` when it does not exist.
fn row_major(data: &[u8], row_bytes: usize, index: usize) -> Option<&[u8]> {
    let start = index.checked_mul(row_bytes).filter(|_| row_bytes > 0)?;
    data.get(start..start.checked_add(row_bytes)?)
}

#[cfg(test)]
#[path = "quant_kernel_tests.rs"]
mod tests;
