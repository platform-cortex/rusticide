// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! What the group kernels' tests share, and the checks every group kernel
//! must pass against its scalar body.

use crate::group::{Q8Blocks, Q8Superblocks, ROWS};

mod body_tests;

pub(crate) use body_tests::group_body_tests;

/// The token every NEON test runs under: the test hosts have `dotprod`.
#[cfg(target_arch = "aarch64")]
pub(crate) fn token() -> crate::Dotprod {
    crate::Dotprod::detect().expect("the test host has dotprod")
}

/// The token every AVX2 test runs under: the test hosts have AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
pub(crate) fn token() -> crate::Avx2 {
    crate::Avx2::detect().expect("the test host has avx2 and fma")
}

/// A random test activation and the view of it a kernel reads.
pub(crate) trait Input: Sized {
    type View<'a>: Copy
    where
        Self: 'a;
    fn random(rng: &mut fastrand::Rng, blocks: usize) -> Self;
    /// `blocks` blocks of `quant` alone.
    fn filled(quant: i8, blocks: usize) -> Self;
    fn view(&self) -> Self::View<'_>;
    /// The view of only its first `blocks` blocks.
    fn first(&self, blocks: usize) -> Self::View<'_>;
}

/// A group kernel taking `I`'s view, for every lifetime of it.
pub(crate) type Kernel<const UNIT: usize, I> =
    for<'a> fn(&[[u8; UNIT]], &[<I as Input>::View<'a>], &mut [[f32; ROWS]]);

/// One token's random `Q8_K` activation, with consistent run sums.
pub(crate) struct Activation {
    pub(crate) scales: Vec<[f32; 8]>,
    pub(crate) quants: Vec<[i8; 256]>,
    pub(crate) sums: Vec<[i32; 8]>,
}

impl Input for Activation {
    type View<'a> = Q8Superblocks<'a>;

    fn random(rng: &mut fastrand::Rng, blocks: usize) -> Self {
        let quants: Vec<[i8; 256]> = (0..blocks)
            .map(|_| core::array::from_fn(|_| rng.i8(-127..)))
            .collect();
        Self::from_quants(quants, (0..blocks).map(|_| rng.f32() * 0.01).collect())
    }

    fn filled(quant: i8, blocks: usize) -> Self {
        Self::from_quants(vec![[quant; 256]; blocks], vec![0.01; blocks])
    }

    fn view(&self) -> Q8Superblocks<'_> {
        Q8Superblocks::new(&self.scales, &self.quants, &self.sums).unwrap()
    }

    fn first(&self, blocks: usize) -> Q8Superblocks<'_> {
        Q8Superblocks::new(&self.scales[..blocks], &self.quants, &self.sums).unwrap()
    }
}

impl Activation {
    pub(crate) fn from_quants(quants: Vec<[i8; 256]>, scales: Vec<f32>) -> Self {
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
        Self {
            scales: scales.into_iter().map(|scale| [scale; 8]).collect(),
            quants,
            sums,
        }
    }
}

/// One token's random `Q8_0` activation.
#[cfg_attr(
    not(target_arch = "aarch64"),
    expect(dead_code, reason = "only the NEON block kernels read one")
)]
pub(crate) struct Blocks {
    pub(crate) scales: Vec<f32>,
    pub(crate) quants: Vec<[i8; 32]>,
}

impl Input for Blocks {
    type View<'a> = Q8Blocks<'a>;

    fn random(rng: &mut fastrand::Rng, blocks: usize) -> Self {
        Self {
            scales: (0..blocks).map(|_| rng.f32() * 0.01).collect(),
            quants: (0..blocks)
                .map(|_| core::array::from_fn(|_| rng.i8(-127..)))
                .collect(),
        }
    }

    fn filled(quant: i8, blocks: usize) -> Self {
        Self {
            scales: vec![0.01; blocks],
            quants: vec![[quant; 32]; blocks],
        }
    }

    fn view(&self) -> Q8Blocks<'_> {
        Q8Blocks::new(&self.scales, &self.quants).unwrap()
    }

    fn first(&self, blocks: usize) -> Q8Blocks<'_> {
        Q8Blocks::new(&self.scales[..blocks], &self.quants).unwrap()
    }
}

pub(crate) fn set_f16(unit: &mut [u8], offset: usize, bits: u16) {
    unit[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
}

/// Random units whose only float scales, the rows' `d` at `D + 2 * row`,
/// are finite f16 values in `[0, 2)`.
pub(crate) fn random_d_units<const UNIT: usize, const D: usize>(
    rng: &mut fastrand::Rng,
    blocks: usize,
) -> Vec<[u8; UNIT]> {
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

/// Each row's `d`, at `D + 2 * row`, gets its special.
pub(crate) fn set_d_specials<const UNIT: usize, const D: usize>(
    unit: &mut [u8; UNIT],
    specials: &[u16; ROWS],
) {
    for (row, &bits) in specials.iter().enumerate() {
        set_f16(unit, D + 2 * row, bits);
    }
}

/// Bitwise equal, or both NaN.
pub(crate) fn same(left: u32, right: u32) -> bool {
    left == right || (f32::from_bits(left).is_nan() && f32::from_bits(right).is_nan())
}

// Miri interprets the scalar bodies roughly 10^4 times slower.
pub(crate) const TOKEN_COUNTS: &[usize] = if cfg!(miri) {
    &[0, 1, 2]
} else {
    &[
        0,
        1,
        2,
        7,
        crate::group::TOKENS + 1,
        3 * crate::group::TOKENS,
    ]
};
pub(crate) const BLOCKS: usize = if cfg!(miri) { 1 } else { 3 };

/// A group kernel under test: its SIMD and scalar bodies, how to make
/// random units with finite float scales, and how to give a unit's rows
/// one special float scale each.
pub(crate) struct Format<const UNIT: usize, I: Input = Activation> {
    pub(crate) kernels: [Kernel<UNIT, I>; 2],
    pub(crate) random_units: fn(&mut fastrand::Rng, usize) -> Vec<[u8; UNIT]>,
    pub(crate) set_specials: fn(&mut [u8; UNIT], &[u16; ROWS]),
}

impl<const UNIT: usize, I: Input> Format<UNIT, I> {
    /// `views` through the SIMD body and the scalar body, as bits.
    pub(crate) fn both(
        &self,
        units: &[[u8; UNIT]],
        views: &[I::View<'_>],
    ) -> [Vec<[u32; ROWS]>; 2] {
        self.kernels.map(|kernel| {
            let mut out = vec![[f32::NAN; ROWS]; views.len()];
            kernel(units, views, &mut out);
            out.into_iter().map(|row| row.map(f32::to_bits)).collect()
        })
    }

    /// `tokens` random activations of `blocks` blocks.
    fn activations(rng: &mut fastrand::Rng, tokens: usize, blocks: usize) -> Vec<I> {
        (0..tokens).map(|_| I::random(rng, blocks)).collect()
    }

    pub(crate) fn the_simd_body_matches_the_scalar_body(&self, seed: u64) {
        let mut rng = fastrand::Rng::with_seed(seed);
        for &tokens in TOKEN_COUNTS {
            let units = (self.random_units)(&mut rng, BLOCKS);
            let activations = Self::activations(&mut rng, tokens, BLOCKS);
            let views: Vec<_> = activations.iter().map(I::view).collect();
            let [native, reference] = self.both(&units, &views);
            assert_eq!(native, reference, "{tokens} tokens");
            let finite = |&value: &u32| f32::from_bits(value).is_finite() && value != 0;
            assert!(native.iter().flatten().all(finite));
        }
    }

    pub(crate) fn every_token_is_computed_as_if_alone(&self, seed: u64) {
        let mut rng = fastrand::Rng::with_seed(seed);
        let units = (self.random_units)(&mut rng, BLOCKS);
        let activations = Self::activations(&mut rng, 3, BLOCKS);
        let views: Vec<_> = activations.iter().map(I::view).collect();
        let [together, _] = self.both(&units, &views);
        for (token, view) in views.iter().enumerate() {
            let [alone, _] = self.both(&units, core::slice::from_ref(view));
            assert_eq!(alone[0], together[token]);
        }
        assert_ne!(together[0], together[1]);
    }

    pub(crate) fn the_shorter_of_group_and_input_bounds_the_blocks(&self, seed: u64) {
        let mut rng = fastrand::Rng::with_seed(seed);
        let units = (self.random_units)(&mut rng, 2);
        let long = I::random(&mut rng, 2);
        let short = long.first(1);
        let [from_short_input, _] = self.both(&units, &[short]);
        let [from_short_group, _] = self.both(&units[..1], &[long.view()]);
        assert_eq!(from_short_input, from_short_group);
        let [nothing, _] = self.both(&[], &[long.view()]);
        assert_eq!(nothing, [[(-0.0_f32).to_bits(); ROWS]]);
    }

    /// Infinities, NaN, subnormals, zeros and the largest half.
    pub(crate) fn special_scales_match_the_scalar_body(&self, seed: u64) {
        let mut rng = fastrand::Rng::with_seed(seed);
        let mut units = (self.random_units)(&mut rng, 1);
        let specials = [
            0x7c00, 0xfc00, 0x7e01, 0x0001, 0x83ff, 0x0000, 0x8000, 0x7bff,
        ];
        (self.set_specials)(&mut units[0], &specials);
        let activation = I::random(&mut rng, 1);
        let [native, reference] = self.both(&units, &[activation.view()]);
        for (&left, &right) in native[0].iter().zip(&reference[0]) {
            assert!(same(left, right), "{left:#x} vs {right:#x}");
        }
    }

    /// Every weight byte `0xff` (finite float scales aside) against the
    /// largest and smallest quants the views accept.
    pub(crate) fn extreme_bytes_match_the_scalar_body(&self) {
        let mut units = vec![[0xff; UNIT]; BLOCKS];
        for unit in &mut units {
            (self.set_specials)(
                unit,
                &[0x7bff, 0x3c00, 0x0001, 0x8001, 0xfbff, 0, 0x3555, 0x8000],
            );
        }
        let activations = [I::filled(127, BLOCKS), I::filled(-127, BLOCKS)];
        let views: Vec<_> = activations.iter().map(I::view).collect();
        let [native, reference] = self.both(&units, &views);
        assert_eq!(native, reference);
    }

    /// Weight data at an odd address: the kernel must not assume alignment.
    pub(crate) fn a_group_at_an_odd_address_gives_the_same_products(&self, seed: u64) {
        let mut rng = fastrand::Rng::with_seed(seed);
        let units = (self.random_units)(&mut rng, 1);
        let mut bytes = vec![0; UNIT + 1];
        bytes[1..].copy_from_slice(units.as_flattened());
        let (shifted, _) = bytes[1..].as_chunks::<UNIT>();
        let activation = I::random(&mut rng, 1);
        let view = [activation.view()];
        assert_eq!(self.both(shifted, &view), self.both(&units, &view));
    }
}
