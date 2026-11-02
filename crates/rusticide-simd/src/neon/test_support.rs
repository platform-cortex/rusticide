// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! What the group kernels' tests share.

use super::group::{GroupKernel, Q8Superblocks, ROWS};

/// One token's random `Q8_K` activation, with consistent run sums.
pub(crate) struct Activation {
    pub(crate) scales: Vec<[f32; 8]>,
    pub(crate) quants: Vec<[i8; 256]>,
    pub(crate) sums: Vec<[i32; 8]>,
}

impl Activation {
    pub(crate) fn random(rng: &mut fastrand::Rng, blocks: usize) -> Self {
        let quants: Vec<[i8; 256]> = (0..blocks)
            .map(|_| core::array::from_fn(|_| rng.i8(..)))
            .collect();
        Self::from_quants(quants, (0..blocks).map(|_| rng.f32() * 0.01).collect())
    }

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

    pub(crate) fn view(&self) -> Q8Superblocks<'_> {
        Q8Superblocks::new(&self.scales, &self.quants, &self.sums)
    }
}

pub(crate) fn set_f16(unit: &mut [u8], offset: usize, bits: u16) {
    unit[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
}

/// `views` through a NEON kernel and its scalar body, as bits.
pub(crate) fn both<const UNIT: usize>(
    kernels: [GroupKernel<UNIT>; 2],
    units: &[[u8; UNIT]],
    views: &[Q8Superblocks<'_>],
) -> [Vec<[u32; ROWS]>; 2] {
    kernels.map(|kernel| {
        let mut out = vec![[f32::NAN; ROWS]; views.len()];
        kernel(units, views, &mut out);
        out.into_iter().map(|row| row.map(f32::to_bits)).collect()
    })
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
        super::group::TOKENS + 1,
        3 * super::group::TOKENS,
    ]
};
pub(crate) const BLOCKS: usize = if cfg!(miri) { 1 } else { 3 };
