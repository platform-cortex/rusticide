// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `Q4_K` units every SIMD body is tested on.

use super::{D, DMIN, UNIT, scalar};
use crate::group::{Q8Superblocks, ROWS};
use crate::test_support::{Activation, BLOCKS, Format, Input, Kernel, set_f16};

/// Random units whose `d` and `dmin` are finite f16 values in `[0, 2)`.
fn random_units(rng: &mut fastrand::Rng, blocks: usize) -> Vec<[u8; UNIT]> {
    (0..blocks)
        .map(|_| {
            let mut unit: [u8; UNIT] = core::array::from_fn(|_| rng.u8(..));
            for row in 0..ROWS {
                set_f16(&mut unit, D + 2 * row, rng.u16(..0x4000));
                set_f16(&mut unit, DMIN + 2 * row, rng.u16(..0x4000));
            }
            unit
        })
        .collect()
}

/// Each row's `d` gets its special and its `dmin` that of the row 3 on.
fn set_specials(unit: &mut [u8; UNIT], specials: &[u16; ROWS]) {
    for (row, &bits) in specials.iter().enumerate() {
        set_f16(unit, D + 2 * row, bits);
        set_f16(unit, DMIN + 2 * row, specials[(row + 3) % ROWS]);
    }
}

/// The checks for the SIMD body `native`, against the scalar body.
pub(crate) const fn format(native: Kernel<UNIT, Activation>) -> Format<UNIT> {
    Format {
        kernels: [native, scalar::group],
        random_units,
        set_specials,
    }
}

/// Run sums at the ends of `i32`, which no activation of real quants
/// holds but the views accept: the min sums wrap in every body alike.
pub(crate) fn wrapping_sums_match_the_scalar_body(format: &Format<UNIT>, seed: u64) {
    let mut rng = fastrand::Rng::with_seed(seed);
    let units = random_units(&mut rng, BLOCKS);
    let mut activation = Activation::random(&mut rng, BLOCKS);
    for (block, sums) in activation.sums.iter_mut().enumerate() {
        *sums = core::array::from_fn(|run| {
            if (block + run) % 2 == 0 {
                i32::MAX
            } else {
                i32::MIN
            }
        });
    }
    let view: Q8Superblocks<'_> = activation.view();
    let [native, reference] = format.both(&units, &[view]);
    assert_eq!(native, reference);
}
