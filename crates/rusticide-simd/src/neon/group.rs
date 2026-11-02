// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! What every eight-row group kernel shares: the group shape and the views
//! of the activations the kernels read.

/// Rows per unit.
pub const ROWS: usize = 8;
/// Inputs a kernel keeps partial sums for at once.
pub const TOKENS: usize = 16;

/// A group kernel: one group's units against each input, the eight rows
/// side by side in the input's slot of the output.
pub type GroupKernel<const UNIT: usize, Input> =
    fn(group: &[[u8; UNIT]], inputs: &[Input], out: &mut [[f32; ROWS]]);

/// One token's `Q8_K` activation as whole 256-element super-blocks: each
/// super-block's scale (stored once per 32 elements, all equal), its
/// quants, and the sums of its eight 32-element runs.
#[derive(Debug, Default, Clone, Copy)]
pub struct Q8Superblocks<'a> {
    pub(crate) scales: &'a [[f32; 8]],
    pub(crate) quants: &'a [[i8; 256]],
    pub(crate) sums: &'a [[i32; 8]],
}

impl<'a> Q8Superblocks<'a> {
    /// A view of the super-blocks all three slices cover.
    #[must_use]
    pub fn new(scales: &'a [[f32; 8]], quants: &'a [[i8; 256]], sums: &'a [[i32; 8]]) -> Self {
        let len = scales.len().min(quants.len()).min(sums.len());
        Self {
            scales: scales.get(..len).unwrap_or_default(),
            quants: quants.get(..len).unwrap_or_default(),
            sums: sums.get(..len).unwrap_or_default(),
        }
    }
}
