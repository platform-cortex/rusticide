// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::hash::Hasher;

/// A multiply-rotate hasher for the merge table's `(u32, u32)` keys. The
/// table is built from the vocabulary and never takes keys from the text
/// being encoded, so `SipHash`'s flooding resistance buys nothing there.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PairHasher(u64);

const SEED: u64 = 0x517c_c1b7_2722_0a95;

impl Hasher for PairHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.write_u32(u32::from(byte));
        }
    }

    fn write_u32(&mut self, value: u32) {
        self.0 = (self.0.rotate_left(5) ^ u64::from(value)).wrapping_mul(SEED);
    }
}

#[cfg(test)]
#[path = "pair_hash_tests.rs"]
mod tests;
