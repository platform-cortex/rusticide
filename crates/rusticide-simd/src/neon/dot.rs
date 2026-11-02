// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Signed 8-bit dot products.

use crate::Dotprod;

/// The dot product of two 32-lane `i8` blocks, summed exactly in `i32`.
#[inline]
#[must_use]
#[cfg_attr(miri, allow(unused_unsafe))]
pub fn dot_i8x32(_token: Dotprod, lhs: &[i8; 32], rhs: &[i8; 32]) -> i32 {
    // SAFETY: a `Dotprod` exists only after `Dotprod::detect` found
    // dotprod on this CPU (NEON is baseline on aarch64), so the CPU running
    // this call has every feature the body enables.
    unsafe { dot(lhs, rhs) }
}

#[cfg(not(miri))]
#[target_feature(enable = "neon,dotprod")]
fn dot(lhs: &[i8; 32], rhs: &[i8; 32]) -> i32 {
    use core::arch::aarch64::{int8x16_t, vaddvq_s32, vdotq_s32, vdupq_n_s32};
    let [lhs_low, lhs_high]: [[i8; 16]; 2] = bytemuck::cast(*lhs);
    let [rhs_low, rhs_high]: [[i8; 16]; 2] = bytemuck::cast(*rhs);
    let lanes = |half: [i8; 16]| -> int8x16_t { bytemuck::cast(half) };
    let sum = vdotq_s32(vdupq_n_s32(0), lanes(lhs_low), lanes(rhs_low));
    vaddvq_s32(vdotq_s32(sum, lanes(lhs_high), lanes(rhs_high)))
}

// Miri cannot execute the intrinsics: this safe stand-in gives the same
// exact integer result, which leaves the wrapper's `unsafe` unused there.
#[cfg(miri)]
fn dot(lhs: &[i8; 32], rhs: &[i8; 32]) -> i32 {
    lhs.iter()
        .zip(rhs)
        .map(|(&left, &right)| i32::from(left) * i32::from(right))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::dot_i8x32;
    use crate::neon::test_support::token;

    fn scalar(lhs: &[i8; 32], rhs: &[i8; 32]) -> i32 {
        lhs.iter()
            .zip(rhs)
            .map(|(&left, &right)| i32::from(left) * i32::from(right))
            .sum()
    }

    fn assert_matches(lhs: &[i8; 32], rhs: &[i8; 32]) {
        assert_eq!(
            dot_i8x32(token(), lhs, rhs),
            scalar(lhs, rhs),
            "{lhs:?} . {rhs:?}"
        );
    }

    // Miri interprets the scalar body roughly 10^4 times slower; the
    // native run covers every position, Miri the two boundary lanes.
    const POSITION_STEP: usize = if cfg!(miri) { 31 } else { 1 };
    const RANDOM_PAIRS: usize = if cfg!(miri) { 50 } else { 1000 };

    #[test]
    fn dot_matches_the_scalar_sum() {
        let ramp: [i8; 32] = core::array::from_fn(|index| {
            i8::try_from(index)
                .unwrap()
                .wrapping_mul(37)
                .wrapping_sub(100)
        });
        for position in (0..32).step_by(POSITION_STEP) {
            for value in i8::MIN..=i8::MAX {
                let mut lhs = [i8::MIN; 32];
                lhs[position] = value;
                assert_matches(&lhs, &[i8::MIN; 32]);
                assert_matches(&lhs, &[i8::MAX; 32]);
                assert_matches(&ramp, &lhs);
            }
        }
        let mut rng = fastrand::Rng::with_seed(0x05ee_dd07);
        for _ in 0..RANDOM_PAIRS {
            let lhs: [i8; 32] = core::array::from_fn(|_| rng.i8(..));
            let rhs: [i8; 32] = core::array::from_fn(|_| rng.i8(..));
            assert_matches(&lhs, &rhs);
        }
    }
}
