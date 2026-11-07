// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The SIMD group kernels' common entry: [`Q8Input`]s viewed as the
//! kernel reads them (views built once per product, see `GroupViews`) and
//! handed to a `rusticide_simd` kernel a chunk of tokens at a time. A token
//! whose view is refused (it holds a `-128`) is computed by the format's
//! portable dot instead, and so is every token when no proof token is held.

use rusticide_simd::group::{GroupKernel, ROWS, TOKENS};

use crate::group_views::SchemeViews;
use crate::q8_input::Q8Input;
use crate::quant_kernel::QuantKernel;

/// One group against every input, into `out[row * inputs.len() + token]`.
/// `views` are those inputs' views; `format` supplies the row reader and
/// the dot for a refused token, or for every token when `proof` (the
/// instruction set's detected token) is `None`.
pub(crate) fn dot_group<Proof: Copy, const UNIT: usize, View: Copy + Default>(
    format: &QuantKernel,
    proof: Option<Proof>,
    kernel: GroupKernel<Proof, UNIT, View>,
    views: &SchemeViews<'_, View>,
    group: &[u8],
    inputs: &[Q8Input],
    out: &mut [f32],
) {
    let Some(proof) = proof else {
        portable_tokens(format, group, inputs, 0..inputs.len(), out);
        return;
    };
    let (units, _) = group.as_chunks::<UNIT>();
    let tokens = inputs.len();
    for (first, chunk) in (0..).step_by(TOKENS).zip(views.views().chunks(TOKENS)) {
        let mut products = [[0.0; ROWS]; TOKENS];
        let count = chunk.len().min(TOKENS);
        kernel(proof, units, chunk, &mut products[..count]);
        for (row, slots) in (0..ROWS).zip(out.chunks_mut(tokens)) {
            for (slot, values) in slots.iter_mut().skip(first).zip(&products[..count]) {
                *slot = values[row];
            }
        }
    }
    if !views.refused().is_empty() {
        portable_tokens(format, group, inputs, views.refused().iter().copied(), out);
    }
}

/// Tokens `tokens` of every row of the group by the format's portable dot,
/// for activations the views refuse or a missing proof.
#[cold]
#[inline(never)]
fn portable_tokens(
    format: &QuantKernel,
    group: &[u8],
    inputs: &[Q8Input],
    tokens: impl Iterator<Item = usize>,
    out: &mut [f32],
) {
    let row_bytes = group.len() / ROWS;
    let stride = inputs.len();
    let mut row = vec![0; row_bytes];
    for (token, input) in tokens.filter_map(|token| Some((token, inputs.get(token)?))) {
        for (index, slots) in out.chunks_mut(stride).enumerate() {
            // A group the adapter prepared always reads: its length is whole
            // rows and `index < ROWS`.
            let read = format.read_row(group, row_bytes, index, &mut row);
            debug_assert!(read, "row {index} of a prepared group must read");
            slots[token] = format.dot_row(&row, input);
        }
    }
}

/// Module `$name` holding a format's group kernel `KERNEL`: the portable
/// `$portable` reading the layout of module `$layout` (`prepare` and
/// `read_row`), whose groups the `rusticide_simd` kernel `$group` computes
/// on the views `GroupViews::$views` picks, under the calling adapter's
/// `proof()`. Its tests are the checks every group kernel passes.
macro_rules! group_kernel {
    ($name:ident: $portable:ident, $layout:path, $group:path, $views:ident) => {
        pub(crate) mod $name {
            use $crate::group_views::GroupViews;
            use $crate::q8_input::Q8Input;
            use $crate::quant_kernel::{Layout, QuantKernel};
            use $layout as layout;

            pub(crate) const KERNEL: QuantKernel = QuantKernel {
                layout: Some(Layout {
                    group_rows: rusticide_simd::group::ROWS,
                    prepare: layout::prepare,
                    read_row: layout::read_row,
                    dot_group,
                }),
                ..$crate::portable::$portable
            };

            fn dot_group(
                group: &[u8],
                inputs: &[Q8Input],
                views: &GroupViews<'_>,
                out: &mut [f32],
            ) {
                dot_group_with(super::proof(), group, inputs, views, out);
            }

            /// The group kernel with `proof`, or the portable kernel without one.
            fn dot_group_with(
                proof: Option<super::Proof>,
                group: &[u8],
                inputs: &[Q8Input],
                views: &GroupViews<'_>,
                out: &mut [f32],
            ) {
                $crate::group_dot::dot_group(
                    &KERNEL,
                    proof,
                    $group,
                    views.$views(),
                    group,
                    inputs,
                    out,
                );
            }

            #[cfg(test)]
            mod tests {
                use super::{KERNEL, dot_group_with};
                use $crate::group_checks;

                #[test]
                fn simd_matches_portable() {
                    group_checks::simd_matches_portable(KERNEL.type_id);
                }

                #[test]
                fn a_batch_is_bit_identical_to_its_tokens_one_at_a_time() {
                    group_checks::a_batch_is_bit_identical_to_its_tokens_one_at_a_time(
                        KERNEL.type_id,
                    );
                }

                #[test]
                fn no_inputs_leave_the_output_untouched() {
                    group_checks::no_inputs_leave_the_output_untouched(KERNEL.type_id);
                }

                #[test]
                fn the_adapter_runs_it_in_groups_of_eight() {
                    group_checks::runs_in_groups_of_eight(KERNEL.type_id);
                }

                #[test]
                fn a_minus_128_activation_matches_portable() {
                    group_checks::a_minus_128_activation_matches_portable(KERNEL.type_id);
                }

                #[test]
                fn wrapper_without_token_matches_portable() {
                    group_checks::wrapper_without_token_matches_portable(
                        KERNEL.type_id,
                        |group, inputs, views, out| dot_group_with(None, group, inputs, views, out),
                    );
                }
            }
        }
    };
}

pub(crate) use group_kernel;
