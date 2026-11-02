// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{D, QH, QL, SCALES, UNIT, q6_k_group, scalar};
use crate::neon::group::{Q8Superblocks, ROWS};
use crate::neon::test_support::{Activation, BLOCKS, TOKEN_COUNTS, same, set_f16};

/// Random units whose `d` are finite f16 values in `[0, 2)`.
fn random_units(rng: &mut fastrand::Rng, blocks: usize) -> Vec<[u8; UNIT]> {
    (0..blocks)
        .map(|_| {
            let mut unit: [u8; UNIT] = core::array::from_fn(|_| rng.u8(..));
            for row in 0..ROWS {
                set_f16(&mut unit, D + 2 * row, rng.u16(..0x4000));
            }
            unit
        })
        .collect()
}

fn both(units: &[[u8; UNIT]], views: &[Q8Superblocks<'_>]) -> [Vec<[u32; ROWS]>; 2] {
    crate::neon::test_support::both([q6_k_group, scalar::group], units, views)
}

#[test]
fn the_neon_body_matches_the_scalar_body() {
    let mut rng = fastrand::Rng::with_seed(0x0006_00a6);
    for &tokens in TOKEN_COUNTS {
        let units = random_units(&mut rng, BLOCKS);
        let activations: Vec<Activation> = (0..tokens)
            .map(|_| Activation::random(&mut rng, BLOCKS))
            .collect();
        let views: Vec<Q8Superblocks<'_>> = activations.iter().map(Activation::view).collect();
        let [native, reference] = both(&units, &views);
        assert_eq!(native, reference, "{tokens} tokens");
        assert!(
            native
                .iter()
                .flatten()
                .all(|&value| f32::from_bits(value).is_finite() && value != 0)
        );
    }
}

#[test]
fn every_token_is_computed_as_if_alone() {
    let mut rng = fastrand::Rng::with_seed(17);
    let units = random_units(&mut rng, BLOCKS);
    let activations: Vec<Activation> = (0..3)
        .map(|_| Activation::random(&mut rng, BLOCKS))
        .collect();
    let views: Vec<Q8Superblocks<'_>> = activations.iter().map(Activation::view).collect();
    let [together, _] = both(&units, &views);
    for (token, view) in views.iter().enumerate() {
        let [alone, _] = both(&units, std::slice::from_ref(view));
        assert_eq!(alone[0], together[token]);
    }
    assert_ne!(together[0], together[1]);
}

#[test]
fn the_shorter_of_group_and_input_bounds_the_blocks() {
    let mut rng = fastrand::Rng::with_seed(19);
    let units = random_units(&mut rng, 2);
    let long = Activation::random(&mut rng, 2);
    let short = Q8Superblocks::new(&long.scales[..1], &long.quants, &long.sums);
    let [from_short_input, _] = both(&units, &[short]);
    let [from_short_group, _] = both(&units[..1], &[long.view()]);
    assert_eq!(from_short_input, from_short_group);
    let [nothing, _] = both(&[], &[long.view()]);
    assert_eq!(nothing, [[(-0.0_f32).to_bits(); ROWS]]);
}

#[test]
fn special_scales_match_the_scalar_body() {
    let mut rng = fastrand::Rng::with_seed(23);
    let mut units = random_units(&mut rng, 1);
    let specials = [
        0x7c00, 0xfc00, 0x7e01, 0x0001, 0x83ff, 0x0000, 0x8000, 0x7bff,
    ];
    for (row, &bits) in specials.iter().enumerate() {
        set_f16(&mut units[0], D + 2 * row, bits);
    }
    let activation = Activation::random(&mut rng, 1);
    let [native, reference] = both(&units, &[activation.view()]);
    for (&left, &right) in native[0].iter().zip(&reference[0]) {
        assert!(same(left, right), "{left:#x} vs {right:#x}");
    }
}

#[test]
fn the_largest_sums_are_exact() {
    let mut unit = [0; UNIT];
    unit[SCALES..D].fill(0x80);
    for row in 0..ROWS {
        set_f16(&mut unit, D + 2 * row, 0x3c00);
    }
    let lowest = Activation::from_quants(vec![[i8::MIN; 256]], vec![1.0]);
    let [native, reference] = both(&[unit], &[lowest.view()]);
    let expected = -134_217_728.0_f32;
    assert_eq!(native, [[expected.to_bits(); ROWS]]);
    assert_eq!(native, reference);
    unit[QL..QH].fill(0xff);
    unit[QH..SCALES].fill(0xff);
    let [native, reference] = both(&[unit], &[lowest.view()]);
    assert_eq!(native, [[(31.0_f32 / 32.0 * -expected).to_bits(); ROWS]]);
    assert_eq!(native, reference);
}

#[test]
fn a_group_at_an_odd_address_gives_the_same_products() {
    let mut rng = fastrand::Rng::with_seed(29);
    let units = random_units(&mut rng, 1);
    let mut bytes = vec![0; UNIT + 1];
    bytes[1..].copy_from_slice(units.as_flattened());
    let (shifted, _) = bytes[1..].as_chunks::<UNIT>();
    let activation = Activation::random(&mut rng, 1);
    assert_eq!(
        both(shifted, &[activation.view()]),
        both(&units, &[activation.view()])
    );
}
