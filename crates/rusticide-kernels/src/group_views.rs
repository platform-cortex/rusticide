// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The group kernels' view of a batch of activations, built once per
//! matrix product: the views refuse a `-128` quant, and scanning for one
//! per group of rows would read every activation once per eight rows.

use std::borrow::Cow;

use rusticide_simd::group::{Q8Blocks, Q8Superblocks, TOKENS};

use crate::q8_input::{Q8Input, Q8Kind};

/// One view per token of each activation scheme; `None` marks a token the
/// view refused.
#[cfg_attr(
    not(target_arch = "aarch64"),
    allow(dead_code, reason = "only the SIMD adapters read the views")
)]
#[derive(Debug, Default)]
pub(crate) struct GroupViews<'a> {
    blocks: Cow<'a, [Option<Q8Blocks<'a>>]>,
    superblocks: Cow<'a, [Option<Q8Superblocks<'a>>]>,
}

#[cfg_attr(
    not(target_arch = "aarch64"),
    allow(dead_code, reason = "only the SIMD adapters read the views")
)]
impl<'a> GroupViews<'a> {
    /// The views of `block` (`Q8_0`) and `super_block` (`Q8_K`) inputs.
    pub(crate) fn new(block: &'a [Q8Input], super_block: &'a [Q8Input]) -> Self {
        Self {
            blocks: block.iter().map(blocks_of).collect(),
            superblocks: super_block.iter().map(superblocks_of).collect(),
        }
    }

    pub(crate) fn blocks(&self) -> &[Option<Q8Blocks<'a>>] {
        &self.blocks
    }

    pub(crate) fn superblocks(&self) -> &[Option<Q8Superblocks<'a>>] {
        &self.superblocks
    }
}

/// Calls `body` with the views of `inputs`, a `kind` batch. Up to
/// [`TOKENS`] inputs are viewed on the stack: a lone product of few
/// tokens costs less than a heap allocation.
pub(crate) fn visit_views<R>(
    inputs: &[Q8Input],
    kind: Q8Kind,
    body: impl FnOnce(&GroupViews<'_>) -> R,
) -> R {
    let count = inputs.len();
    if count > TOKENS {
        return match kind {
            Q8Kind::Block => body(&GroupViews::new(inputs, &[])),
            Q8Kind::SuperBlock => body(&GroupViews::new(&[], inputs)),
        };
    }
    match kind {
        Q8Kind::Block => {
            let blocks = stacked(inputs, blocks_of);
            body(&GroupViews {
                blocks: Cow::Borrowed(&blocks[..count]),
                superblocks: Cow::Borrowed(&[]),
            })
        }
        Q8Kind::SuperBlock => {
            let superblocks = stacked(inputs, superblocks_of);
            body(&GroupViews {
                blocks: Cow::Borrowed(&[]),
                superblocks: Cow::Borrowed(&superblocks[..count]),
            })
        }
    }
}

/// The views of the first [`TOKENS`] inputs, the rest `None`.
fn stacked<'i, View: Copy>(
    inputs: &'i [Q8Input],
    view: fn(&'i Q8Input) -> Option<View>,
) -> [Option<View>; TOKENS] {
    let mut views = [None; TOKENS];
    for (slot, input) in views.iter_mut().zip(inputs) {
        *slot = view(input);
    }
    views
}

fn blocks_of(input: &Q8Input) -> Option<Q8Blocks<'_>> {
    Q8Blocks::new(&input.scales, input.quants.as_chunks().0)
}

fn superblocks_of(input: &Q8Input) -> Option<Q8Superblocks<'_>> {
    Q8Superblocks::new(
        input.scales.as_chunks().0,
        input.quants.as_chunks().0,
        input.sums.as_chunks().0,
    )
}
