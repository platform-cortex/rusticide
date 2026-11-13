// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Activations quantized to signed bytes, the right-hand side of the integer
//! dot kernels.

use crate::adapter::active;

/// Elements per activation block.
pub const Q8_BLOCK: usize = 32;

/// An `f32` vector quantized in blocks of [`Q8_BLOCK`]: each block keeps one
/// scale, its values rounded to `i8`, and the sum of those `i8` values.
#[derive(Debug, Default, Clone)]
pub struct Q8Input {
    /// The scheme of the last [`Self::quantize`], which sets it; it tells
    /// [`Self::dequantize`] how to read the scales.
    pub(crate) kind: Q8Kind,
    pub(crate) scales: Vec<f32>,
    pub(crate) quants: Vec<i8>,
    pub(crate) sums: Vec<i32>,
}

/// The `K` [`Q8Input`] blocks covering one weight block.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Span<'a, const K: usize> {
    pub(crate) scales: &'a [f32; K],
    pub(crate) quants: &'a [[i8; Q8_BLOCK]; K],
    pub(crate) sums: &'a [i32; K],
}

/// How [`Q8Input::quantize`] scales the activation, after ggml: matching the
/// scheme llama.cpp's CPU backend pairs with each weight format keeps greedy
/// decoding on the same path as llama.cpp.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Q8Kind {
    /// ggml `Q8_0`: one f16 scale `max|x| / 127` per 32 elements.
    #[default]
    Block,
    /// ggml `Q8_K`: one f32 scale per 256 elements, signed so that the
    /// largest-magnitude element maps to -127.
    SuperBlock,
    /// bitnet.cpp's legacy `I8_S`: one f32 divisor `127 / max|x|` for the
    /// whole row, its `max` taken in f64 from `1e-5`, stored once per 32
    /// elements; quants span `[-128, 127]`, so no group kernel reads them.
    /// The `max` covers the whole blocks only; the C takes it over all `n`
    /// elements, the same value for `BitNet`'s widths (multiples of 32).
    Row,
}

impl Q8Input {
    /// Quantizes `input` as `kind`, reusing this value's buffers. A trailing
    /// partial block is dropped; callers pass whole blocks.
    pub fn quantize(&mut self, input: &[f32], kind: Q8Kind) {
        self.kind = kind;
        (active().quantize)(self, input, kind);
    }

    /// The number of quantized elements.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.quants.len()
    }

    /// Whether no element is quantized.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.quants.is_empty()
    }

    /// The decoded values, `scale * quant` (`quant / scale` for a
    /// [`Q8Kind::Row`]), for tests and reference checks.
    #[must_use]
    pub fn dequantize(&self) -> Vec<f32> {
        let row = self.kind == Q8Kind::Row;
        self.quants
            .as_chunks::<Q8_BLOCK>()
            .0
            .iter()
            .zip(&self.scales)
            .flat_map(|(quants, &scale)| {
                quants.iter().map(move |&q| {
                    if row {
                        f32::from(q) / scale
                    } else {
                        scale * f32::from(q)
                    }
                })
            })
            .collect()
    }

    /// Consecutive runs of `K` blocks, one per weight block.
    pub(crate) fn spans<const K: usize>(&self) -> impl Iterator<Item = Span<'_, K>> {
        self.spans_from(0)
    }

    /// [`Self::spans`] from the span of weight block `first` on.
    pub(crate) fn spans_from<const K: usize>(
        &self,
        first: usize,
    ) -> impl Iterator<Item = Span<'_, K>> {
        let (blocks, _) = self.quants.as_chunks::<Q8_BLOCK>();
        let (quants, _) = blocks.as_chunks::<K>();
        let (scales, _) = self.scales.as_chunks::<K>();
        let (sums, _) = self.sums.as_chunks::<K>();
        let tail = |len: usize| first.min(len)..;
        scales[tail(scales.len())]
            .iter()
            .zip(&quants[tail(quants.len())])
            .zip(&sums[tail(sums.len())])
            .map(|((scales, quants), sums)| Span {
                scales,
                quants,
                sums,
            })
    }
}

#[cfg(test)]
#[path = "q8_input_tests.rs"]
mod tests;
