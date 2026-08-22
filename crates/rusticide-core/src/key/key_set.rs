// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{BackendComponent, DispatchKey, NUM_BACKENDS};

/// Every `DispatchKey` variant, in discriminant order.
///
/// Single source of truth for mapping a bit index back to a variant in
/// [`DispatchKeySet::highest_functionality`], and (via `key::kernel_table`)
/// for the kernel-table slot layout. A `match` on the raw `u8` index cannot
/// be checked for exhaustiveness against the enum, so a variant appended to
/// `DispatchKey` and forgotten here would silently dispatch to no kernel;
/// walking this array instead turns that into "not registered here", caught
/// by the round-trip test below. Update in order when `DispatchKey` gains a
/// variant.
pub(super) const ALL_FUNCTIONALITIES: [DispatchKey; 3] = [
    DispatchKey::Dense,
    DispatchKey::Functionalize,
    DispatchKey::Autograd,
];

/// Fails to compile if a `DispatchKey` variant is missing from
/// `ALL_FUNCTIONALITIES`. The match is exhaustive over the *type*, so rustc
/// (not a test) rejects it the moment a variant is added here without also
/// being listed above -- the guarantee `ALL_FUNCTIONALITIES.into_iter().find`
/// cannot give on its own, since a lookup miss is just `None`. Never add a
/// `_` arm: that would silence exactly the check this exists for.
#[allow(dead_code)]
const fn assert_all_functionalities_registered(k: DispatchKey) {
    match k {
        DispatchKey::Dense | DispatchKey::Functionalize | DispatchKey::Autograd => (),
    }
}

/// A set of dispatch keys packed into one `u64`.
///
/// The low [`NUM_BACKENDS`] bits hold backends; the remaining bits hold
/// functionalities. Storing them orthogonally avoids giving every
/// (backend x functionality) pair its own bit -- that would grow
/// quadratically. The runtime key is instead reconstituted by taking the
/// highest set bit of each half independently, mirroring
/// `c10/core/DispatchKeySet.h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DispatchKeySet(u64);

impl DispatchKeySet {
    /// A keyset with no backend and no functionality set.
    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Returns a copy of this set with `b`'s bit set.
    #[must_use]
    pub const fn with_backend(self, b: BackendComponent) -> Self {
        Self(self.0 | (1u64 << (b as u8 - 1)))
    }

    /// Returns a copy of this set with `k`'s bit set.
    #[must_use]
    pub const fn with_functionality(self, k: DispatchKey) -> Self {
        Self(self.0 | (1u64 << (NUM_BACKENDS + k as u8)))
    }

    /// Multiple dispatch: OR the keysets of every operand together.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether `b`'s bit is set.
    #[must_use]
    pub const fn has_backend(self, b: BackendComponent) -> bool {
        self.0 & (1u64 << (b as u8 - 1)) != 0
    }

    /// Whether this set has no backend and no functionality set.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The highest-priority (numerically greatest) functionality in the set,
    /// or `None` if no functionality bit is set.
    #[must_use]
    pub fn highest_functionality(self) -> Option<DispatchKey> {
        let bits = self.0 >> NUM_BACKENDS;
        if bits == 0 {
            return None;
        }
        // `bits.ilog2()` is the 0-based index of the highest set bit
        // (equivalent to `63 - leading_zeros()`, but clippy's `manual_ilog2`
        // wants the named method). `DispatchKey` discriminants start at 0 at
        // exactly that bit position, so `idx` IS the discriminant.
        let idx = bits.ilog2();
        ALL_FUNCTIONALITIES
            .into_iter()
            .find(|k| u32::from(*k as u8) == idx)
    }

    /// The highest-priority (numerically greatest) backend in the set, or
    /// `None` if no backend bit is set.
    #[must_use]
    pub const fn highest_backend(self) -> Option<BackendComponent> {
        let mask = (1u64 << NUM_BACKENDS) - 1;
        let bits = self.0 & mask;
        if bits == 0 {
            return None;
        }
        // As in `highest_functionality`, `bits.ilog2()` finds the highest set
        // bit's index. `BackendComponent` discriminants start at 1 (0 means
        // "no backend"), one more than the bit position, hence
        // `ilog2() + 1 == 64 - leading_zeros()`.
        match 64 - bits.leading_zeros() {
            1 => Some(BackendComponent::Cpu),
            2 => Some(BackendComponent::Gpu),
            _ => None,
        }
    }

    /// The raw packed representation.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

// `usize::from` isn't const-stable at this MSRV; `as usize` here is lossless
// since `usize` is always at least as wide as `u8`.
const _: () = assert!(ALL_FUNCTIONALITIES.len() + NUM_BACKENDS as usize <= 64);

#[cfg(test)]
mod tests {
    use super::DispatchKeySet;
    use crate::{BackendComponent, DispatchKey};

    #[test]
    fn an_empty_set_has_no_keys() {
        let s = DispatchKeySet::empty();
        assert!(s.highest_functionality().is_none());
        assert!(s.highest_backend().is_none());
    }

    #[test]
    fn a_set_reports_the_backend_it_was_given() {
        let s = DispatchKeySet::empty().with_backend(BackendComponent::Cpu);
        assert!(s.has_backend(BackendComponent::Cpu));
        assert!(!s.has_backend(BackendComponent::Gpu));
        assert_eq!(s.highest_backend(), Some(BackendComponent::Cpu));
    }

    #[test]
    fn the_highest_functionality_wins_regardless_of_insertion_order() {
        let a = DispatchKeySet::empty()
            .with_functionality(DispatchKey::Dense)
            .with_functionality(DispatchKey::Autograd);
        let b = DispatchKeySet::empty()
            .with_functionality(DispatchKey::Autograd)
            .with_functionality(DispatchKey::Dense);
        assert_eq!(a.highest_functionality(), Some(DispatchKey::Autograd));
        assert_eq!(b.highest_functionality(), Some(DispatchKey::Autograd));
    }

    #[test]
    fn union_is_how_multiple_dispatch_combines_operands() {
        let cpu_dense = DispatchKeySet::empty()
            .with_backend(BackendComponent::Cpu)
            .with_functionality(DispatchKey::Dense);
        let autograd = DispatchKeySet::empty().with_functionality(DispatchKey::Autograd);
        let merged = cpu_dense.union(autograd);
        assert_eq!(merged.highest_functionality(), Some(DispatchKey::Autograd));
        assert_eq!(merged.highest_backend(), Some(BackendComponent::Cpu));
    }

    /// The whole reason backends and functionalities are split into two
    /// orthogonal halves is so multiple dispatch can carry more than one
    /// backend bit at once; without this test, `highest_backend`'s
    /// priority-resolution branch (as opposed to its trivial single-bit
    /// path) is never exercised.
    #[test]
    fn union_resolves_the_higher_priority_backend_across_devices() {
        let cpu = DispatchKeySet::empty().with_backend(BackendComponent::Cpu);
        let gpu = DispatchKeySet::empty().with_backend(BackendComponent::Gpu);
        let merged = cpu.union(gpu);
        assert_eq!(merged.highest_backend(), Some(BackendComponent::Gpu));
        assert!(merged.has_backend(BackendComponent::Cpu));
        assert!(merged.has_backend(BackendComponent::Gpu));
    }

    /// Guards the append-a-variant hazard: this walks `ALL_FUNCTIONALITIES`
    /// rather than trusting a hand-written match, so every variant actually
    /// registered here is proven to round-trip through the bitset.
    #[test]
    fn every_registered_functionality_round_trips_through_the_bitset() {
        for k in super::ALL_FUNCTIONALITIES {
            let s = DispatchKeySet::empty().with_functionality(k);
            assert_eq!(s.highest_functionality(), Some(k));
        }
    }

    /// Checks discriminants are `0..len` **in order**, unsorted: the kernel
    /// table (`key::kernel_table`) accumulates offsets by walking
    /// `ALL_FUNCTIONALITIES` and breaking on the first entry that outranks
    /// the target, which is only correct if the array is already in
    /// ascending discriminant order. Sorting before comparing (as an earlier
    /// version of this test did) would pass even if the array were listed
    /// out of order.
    #[test]
    fn all_functionalities_covers_every_discriminant_in_order() {
        let discriminants: Vec<u8> = super::ALL_FUNCTIONALITIES
            .iter()
            .map(|k| *k as u8)
            .collect();
        let expected: Vec<u8> =
            (0..u8::try_from(super::ALL_FUNCTIONALITIES.len()).expect("small")).collect();
        assert_eq!(discriminants, expected);
    }
}
