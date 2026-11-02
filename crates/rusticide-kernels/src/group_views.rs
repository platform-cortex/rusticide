// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The group kernels' view of a batch of activations, built once per
//! matrix product: the views refuse a `-128` quant, and scanning for one
//! per group of rows would read every activation once per eight rows.

use std::borrow::Cow;

use rusticide_simd::group::{Q8Blocks, Q8Superblocks, TOKENS};

use crate::q8_input::{Q8Input, Q8Kind};

/// One scheme's views, ready for a kernel to read in place: `views[token]`
/// for every token, a refused token holding the empty default view, and
/// the refused tokens in ascending order.
#[cfg_attr(
    not(target_arch = "aarch64"),
    allow(dead_code, reason = "only the SIMD adapters read the views")
)]
#[derive(Debug, Default)]
pub(crate) struct SchemeViews<'a, View: Clone> {
    views: Cow<'a, [View]>,
    refused: Cow<'a, [usize]>,
}

#[cfg_attr(
    not(target_arch = "aarch64"),
    allow(dead_code, reason = "only the SIMD adapters read the views")
)]
impl<'a, View: Copy + Default> SchemeViews<'a, View> {
    fn new<'i: 'a>(inputs: &'i [Q8Input], view: fn(&'i Q8Input) -> Option<View>) -> Self {
        let mut refused = Vec::new();
        let views = (0..)
            .zip(inputs)
            .map(|(token, input)| {
                view(input).unwrap_or_else(|| {
                    refused.push(token);
                    View::default()
                })
            })
            .collect();
        Self {
            views: Cow::Owned(views),
            refused: Cow::Owned(refused),
        }
    }

    pub(crate) fn views(&self) -> &[View] {
        &self.views
    }

    pub(crate) fn refused(&self) -> &[usize] {
        &self.refused
    }
}

/// One view per token of each activation scheme.
#[derive(Debug, Default)]
pub(crate) struct GroupViews<'a> {
    blocks: SchemeViews<'a, Q8Blocks<'a>>,
    superblocks: SchemeViews<'a, Q8Superblocks<'a>>,
}

#[cfg_attr(
    not(target_arch = "aarch64"),
    allow(dead_code, reason = "only the SIMD adapters read the views")
)]
impl<'a> GroupViews<'a> {
    /// The views of `block` (`Q8_0`) and `super_block` (`Q8_K`) inputs.
    pub(crate) fn new(block: &'a [Q8Input], super_block: &'a [Q8Input]) -> Self {
        Self {
            blocks: SchemeViews::new(block, blocks_of),
            superblocks: SchemeViews::new(super_block, superblocks_of),
        }
    }

    pub(crate) const fn blocks(&self) -> &SchemeViews<'a, Q8Blocks<'a>> {
        &self.blocks
    }

    pub(crate) const fn superblocks(&self) -> &SchemeViews<'a, Q8Superblocks<'a>> {
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
    if inputs.len() > TOKENS {
        return match kind {
            Q8Kind::Block => body(&GroupViews::new(inputs, &[])),
            Q8Kind::SuperBlock => body(&GroupViews::new(&[], inputs)),
        };
    }
    match kind {
        Q8Kind::Block => {
            let stack = Stacked::new(inputs, blocks_of);
            body(&GroupViews {
                blocks: stack.borrow(),
                superblocks: SchemeViews::default(),
            })
        }
        Q8Kind::SuperBlock => {
            let stack = Stacked::new(inputs, superblocks_of);
            body(&GroupViews {
                blocks: SchemeViews::default(),
                superblocks: stack.borrow(),
            })
        }
    }
}

/// [`SchemeViews`] of at most [`TOKENS`] inputs, on the stack.
struct Stacked<View> {
    views: [View; TOKENS],
    count: usize,
    refused: [usize; TOKENS],
    refusals: usize,
}

impl<View: Copy + Default> Stacked<View> {
    fn new<'i>(inputs: &'i [Q8Input], view: fn(&'i Q8Input) -> Option<View>) -> Self {
        let mut stack = Self {
            views: [View::default(); TOKENS],
            count: 0,
            refused: [0; TOKENS],
            refusals: 0,
        };
        for (slot, input) in stack.views.iter_mut().zip(inputs) {
            if let Some(found) = view(input) {
                *slot = found;
            } else if let Some(refused) = stack.refused.get_mut(stack.refusals) {
                *refused = stack.count;
                stack.refusals += 1;
            }
            stack.count += 1;
        }
        stack
    }

    fn borrow(&self) -> SchemeViews<'_, View> {
        SchemeViews {
            views: Cow::Borrowed(self.views.get(..self.count).unwrap_or_default()),
            refused: Cow::Borrowed(self.refused.get(..self.refusals).unwrap_or_default()),
        }
    }
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

#[cfg(test)]
#[path = "group_views_tests.rs"]
mod tests;
