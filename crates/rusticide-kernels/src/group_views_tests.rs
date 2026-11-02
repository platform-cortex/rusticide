// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_simd::group::TOKENS;

use super::{GroupViews, visit_views};
use crate::q8_input::{Q8Input, Q8Kind};
use crate::test_support::pseudo_random_values;

/// `tokens` inputs of `kind`, a `-128` quant in each token of `refused`.
fn inputs(kind: Q8Kind, tokens: usize, refused: &[usize]) -> Vec<Q8Input> {
    (0..tokens)
        .map(|token| {
            let mut input = Q8Input::default();
            input.quantize(&pseudo_random_values(512, 9), kind);
            if refused.contains(&token) {
                input.quants[300] = i8::MIN;
            }
            input
        })
        .collect()
}

#[test]
fn block_views_list_their_refused_tokens_in_order() {
    let refused = [1, TOKENS + 4];
    let inputs = inputs(Q8Kind::Block, 2 * TOKENS + 3, &refused);
    let views = GroupViews::new(&inputs, &[]);
    assert_eq!(views.blocks().views().len(), inputs.len());
    assert_eq!(views.blocks().refused(), refused);
    assert!(views.superblocks().views().is_empty());
}

#[test]
fn superblock_views_list_their_refused_tokens_in_order() {
    let refused = [0, 2 * TOKENS];
    let inputs = inputs(Q8Kind::SuperBlock, 2 * TOKENS + 1, &refused);
    let views = GroupViews::new(&[], &inputs);
    assert_eq!(views.superblocks().views().len(), inputs.len());
    assert_eq!(views.superblocks().refused(), refused);
    assert!(views.blocks().views().is_empty());
}

#[test]
fn few_inputs_are_viewed_on_the_stack_with_their_refusals() {
    for kind in [Q8Kind::Block, Q8Kind::SuperBlock] {
        let inputs = inputs(kind, 3, &[2]);
        let (len, refused) = visit_views(&inputs, kind, |views| match kind {
            Q8Kind::Block => (
                views.blocks().views().len(),
                views.blocks().refused().to_vec(),
            ),
            Q8Kind::SuperBlock => (
                views.superblocks().views().len(),
                views.superblocks().refused().to_vec(),
            ),
        });
        assert_eq!((len, refused), (3, vec![2]));
    }
}

#[test]
fn many_inputs_are_viewed_on_the_heap_with_their_refusals() {
    for kind in [Q8Kind::Block, Q8Kind::SuperBlock] {
        let inputs = inputs(kind, TOKENS + 1, &[TOKENS]);
        let refused = visit_views(&inputs, kind, |views| match kind {
            Q8Kind::Block => views.blocks().refused().to_vec(),
            Q8Kind::SuperBlock => views.superblocks().refused().to_vec(),
        });
        assert_eq!(refused, [TOKENS]);
    }
}
