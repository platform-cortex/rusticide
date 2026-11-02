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

/// Whether any quant is `-128`. The views refuse such an activation so a
/// kernel may negate any quant it reads (the AVX2 sign trick) without
/// overflowing `i8`.
fn has_minus_128(quants: &[i8]) -> bool {
    quants.contains(&i8::MIN)
}

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
    /// A view of the super-blocks all three slices cover; `None` when a
    /// covered quant is `-128`, which `vpmaddubsw`-style kernels cannot
    /// take.
    #[must_use]
    pub fn new(
        scales: &'a [[f32; 8]],
        quants: &'a [[i8; 256]],
        sums: &'a [[i32; 8]],
    ) -> Option<Self> {
        let len = scales.len().min(quants.len()).min(sums.len());
        let quants = quants.get(..len).unwrap_or_default();
        if has_minus_128(quants.as_flattened()) {
            return None;
        }
        Some(Self {
            scales: scales.get(..len).unwrap_or_default(),
            quants,
            sums: sums.get(..len).unwrap_or_default(),
        })
    }
}

/// One token's `Q8_0` activation as whole 32-element blocks: each block's
/// scale and its quants.
#[derive(Debug, Default, Clone, Copy)]
pub struct Q8Blocks<'a> {
    pub(crate) scales: &'a [f32],
    pub(crate) quants: &'a [[i8; 32]],
}

impl<'a> Q8Blocks<'a> {
    /// A view of the blocks both slices cover; `None` when a covered quant
    /// is `-128`, which `vpmaddubsw`-style kernels cannot take.
    #[must_use]
    pub fn new(scales: &'a [f32], quants: &'a [[i8; 32]]) -> Option<Self> {
        let len = scales.len().min(quants.len());
        let quants = quants.get(..len).unwrap_or_default();
        if has_minus_128(quants.as_flattened()) {
            return None;
        }
        Some(Self {
            scales: scales.get(..len).unwrap_or_default(),
            quants,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{Q8Blocks, Q8Superblocks};

    #[test]
    fn blocks_refuse_minus_128() {
        let mut quants = [[1_i8; 32]; 2];
        quants[1][31] = i8::MIN;
        assert!(Q8Blocks::new(&[1.0, 1.0], &quants).is_none());
        assert!(Q8Blocks::new(&[1.0], &quants).is_some());
    }

    #[test]
    fn superblocks_refuse_minus_128() {
        let mut quants = [[1_i8; 256]; 2];
        quants[1][255] = i8::MIN;
        let (scales, sums) = ([[1.0; 8]; 2], [[0; 8]; 2]);
        assert!(Q8Superblocks::new(&scales, &quants, &sums).is_none());
        assert!(Q8Superblocks::new(&scales, &quants, &sums[..1]).is_some());
    }

    #[test]
    fn views_accept_minus_127() {
        let blocks = Q8Blocks::new(&[1.0], &[[-127; 32]]).unwrap();
        assert_eq!((blocks.scales.len(), blocks.quants.len()), (1, 1));
        let (scales, sums) = ([[1.0; 8]], [[0; 8]]);
        let view = Q8Superblocks::new(&scales, &[[-127; 256]], &sums).unwrap();
        assert_eq!((view.scales.len(), view.quants.len()), (1, 1));
    }
}
