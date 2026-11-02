// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{D, DMIN, Q8Superblocks, ROWS, TOKENS, UNIT, q4_k_group, scalar};

struct Activation {
    scales: Vec<[f32; 8]>,
    quants: Vec<[i8; 256]>,
    sums: Vec<[i32; 8]>,
}

impl Activation {
    fn random(rng: &mut fastrand::Rng, blocks: usize) -> Self {
        let quants: Vec<[i8; 256]> = (0..blocks)
            .map(|_| core::array::from_fn(|_| rng.i8(..)))
            .collect();
        let sums = quants
            .iter()
            .map(|block| {
                core::array::from_fn(|run| {
                    block[32 * run..32 * run + 32]
                        .iter()
                        .map(|&q| i32::from(q))
                        .sum()
                })
            })
            .collect();
        let scales = (0..blocks).map(|_| [rng.f32() * 0.01; 8]).collect();
        Self {
            scales,
            quants,
            sums,
        }
    }

    fn view(&self) -> Q8Superblocks<'_> {
        Q8Superblocks::new(&self.scales, &self.quants, &self.sums)
    }
}

fn set_f16(unit: &mut [u8; UNIT], offset: usize, bits: u16) {
    unit[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
}

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

fn both(units: &[[u8; UNIT]], views: &[Q8Superblocks<'_>]) -> (Vec<[u32; 8]>, Vec<[u32; 8]>) {
    let mut native = vec![[f32::NAN; ROWS]; views.len()];
    let mut reference = native.clone();
    q4_k_group(units, views, &mut native);
    scalar::group(units, views, &mut reference);
    let bits = |rows: Vec<[f32; ROWS]>| -> Vec<[u32; 8]> {
        rows.into_iter().map(|row| row.map(f32::to_bits)).collect()
    };
    (bits(native), bits(reference))
}

// Miri interprets the scalar body roughly 10^4 times slower.
const TOKEN_COUNTS: &[usize] = if cfg!(miri) {
    &[0, 1, 2]
} else {
    &[0, 1, 2, 7, TOKENS + 1, 3 * TOKENS]
};
const BLOCKS: usize = if cfg!(miri) { 1 } else { 3 };

#[test]
fn the_neon_body_matches_the_scalar_body() {
    let mut rng = fastrand::Rng::with_seed(0x0004_00a4);
    for &tokens in TOKEN_COUNTS {
        let units = random_units(&mut rng, BLOCKS);
        let activations: Vec<Activation> = (0..tokens)
            .map(|_| Activation::random(&mut rng, BLOCKS))
            .collect();
        let views: Vec<Q8Superblocks<'_>> = activations.iter().map(Activation::view).collect();
        let (native, reference) = both(&units, &views);
        assert_eq!(native, reference, "{tokens} tokens");
        assert!(
            native
                .iter()
                .flatten()
                .all(|&value| f32::from_bits(value).is_finite())
        );
    }
}

#[test]
fn every_token_is_computed_as_if_alone() {
    let mut rng = fastrand::Rng::with_seed(7);
    let units = random_units(&mut rng, BLOCKS);
    let activations: Vec<Activation> = (0..3)
        .map(|_| Activation::random(&mut rng, BLOCKS))
        .collect();
    let views: Vec<Q8Superblocks<'_>> = activations.iter().map(Activation::view).collect();
    let (together, _) = both(&units, &views);
    for (token, view) in views.iter().enumerate() {
        assert_eq!(
            both(&units, std::slice::from_ref(view)).0[0],
            together[token]
        );
    }
    assert_ne!(together[0], together[1]);
}

#[test]
fn the_shorter_of_group_and_input_bounds_the_blocks() {
    let mut rng = fastrand::Rng::with_seed(11);
    let units = random_units(&mut rng, 2);
    let long = Activation::random(&mut rng, 2);
    let short = Q8Superblocks::new(&long.scales[..1], &long.quants, &long.sums);
    let (from_short_input, _) = both(&units, &[short]);
    let (from_short_group, _) = both(&units[..1], &[long.view()]);
    assert_eq!(from_short_input, from_short_group);
    let (nothing, _) = both(&[], &[long.view()]);
    assert_eq!(nothing, [[(-0.0_f32).to_bits(); 8]]);
}

#[test]
fn special_scales_match_the_scalar_body() {
    let mut rng = fastrand::Rng::with_seed(13);
    let mut units = random_units(&mut rng, 1);
    let specials = [
        0x7c00, 0xfc00, 0x7e01, 0x0001, 0x83ff, 0x0000, 0x8000, 0x7bff,
    ];
    for (row, &bits) in specials.iter().enumerate() {
        set_f16(&mut units[0], D + 2 * row, bits);
        set_f16(&mut units[0], DMIN + 2 * row, specials[(row + 3) % ROWS]);
    }
    let activation = Activation::random(&mut rng, 1);
    let (native, reference) = both(&units, &[activation.view()]);
    for (left, right) in native[0].iter().zip(&reference[0]) {
        let (left, right) = (f32::from_bits(*left), f32::from_bits(*right));
        assert!(
            left.to_bits() == right.to_bits() || (left.is_nan() && right.is_nan()),
            "{left} vs {right}"
        );
    }
}

// `half` converts with inline assembly on this target, which Miri cannot run.
#[test]
#[cfg_attr(miri, ignore)]
fn f16_widening_matches_half_for_every_value() {
    for bits in 0..=u16::MAX {
        let expected = half::f16::from_bits(bits).to_f32();
        let got = scalar::f16_to_f32(bits);
        assert!(
            got.to_bits() == expected.to_bits() || (got.is_nan() && expected.is_nan()),
            "{bits:#06x}"
        );
    }
}
