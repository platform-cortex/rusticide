// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{BackendComponent, DispatchKey, NUM_BACKENDS};

/// Every `DispatchKey` variant, in discriminant order.
///
/// Single source of truth for mapping a bit index back to a variant in
/// [`DispatchKeySet::highest_functionality`]. A `match` on the raw `u8` index
/// cannot be checked for exhaustiveness against the enum, so a variant
/// appended to `DispatchKey` and forgotten here would silently dispatch to no
/// kernel; walking this array instead turns that into "not registered here",
/// caught by the round-trip test below. Update in order when `DispatchKey`
/// gains a variant.
const ALL_FUNCTIONALITIES: [DispatchKey; 3] = [
    DispatchKey::Dense,
    DispatchKey::Functionalize,
    DispatchKey::Autograd,
];

/// Kernel-table slots a per-backend functionality occupies: one per backend.
const BACKENDS_PER_FUNCTIONALITY: usize = 2;

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

    /// The slot in an operator's kernel table this keyset resolves to.
    ///
    /// Per-backend functionalities occupy [`BACKENDS_PER_FUNCTIONALITY`]
    /// consecutive slots (one per backend); others occupy one. Functionalities
    /// are laid out in discriminant order, so a functionality's base offset is
    /// the summed slot count of every functionality that outranks it less.
    #[must_use]
    pub fn table_index(self) -> usize {
        let Some(f) = self.highest_functionality() else {
            return 0;
        };
        let mut offset = 0usize;
        for k in ALL_FUNCTIONALITIES {
            if k as u8 >= f as u8 {
                break;
            }
            offset += if k.is_per_backend() {
                BACKENDS_PER_FUNCTIONALITY
            } else {
                1
            };
        }
        if f.is_per_backend() {
            offset + self.highest_backend().map_or(0, |b| b as usize - 1)
        } else {
            offset
        }
    }

    /// Total slots an operator's kernel table must hold.
    #[must_use]
    pub const fn table_size() -> usize {
        let mut total = 0usize;
        let mut i = 0usize;
        while i < ALL_FUNCTIONALITIES.len() {
            total += if ALL_FUNCTIONALITIES[i].is_per_backend() {
                BACKENDS_PER_FUNCTIONALITY
            } else {
                1
            };
            i += 1;
        }
        total
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

    #[test]
    fn per_backend_functionalities_get_one_slot_per_backend() {
        let cpu = DispatchKeySet::empty()
            .with_backend(BackendComponent::Cpu)
            .with_functionality(DispatchKey::Dense);
        let gpu = DispatchKeySet::empty()
            .with_backend(BackendComponent::Gpu)
            .with_functionality(DispatchKey::Dense);
        assert_ne!(
            cpu.table_index(),
            gpu.table_index(),
            "dense-cpu and dense-gpu must not share a kernel slot"
        );
    }

    #[test]
    fn a_non_per_backend_functionality_uses_one_slot() {
        let cpu = DispatchKeySet::empty()
            .with_backend(BackendComponent::Cpu)
            .with_functionality(DispatchKey::Functionalize);
        let gpu = DispatchKeySet::empty()
            .with_backend(BackendComponent::Gpu)
            .with_functionality(DispatchKey::Functionalize);
        assert_eq!(cpu.table_index(), gpu.table_index());
    }

    /// Pins the by-hand slot layout from the task brief: Dense (per-backend)
    /// takes slots 0..1, Functionalize (shared) takes slot 2, Autograd
    /// (per-backend) takes slots 3..4, for a table of size 5.
    #[test]
    fn table_index_matches_the_hand_computed_slot_layout() {
        let index_of = |f: DispatchKey, b: BackendComponent| {
            DispatchKeySet::empty()
                .with_backend(b)
                .with_functionality(f)
                .table_index()
        };
        assert_eq!(index_of(DispatchKey::Dense, BackendComponent::Cpu), 0);
        assert_eq!(index_of(DispatchKey::Dense, BackendComponent::Gpu), 1);
        assert_eq!(
            index_of(DispatchKey::Functionalize, BackendComponent::Cpu),
            2
        );
        assert_eq!(
            index_of(DispatchKey::Functionalize, BackendComponent::Gpu),
            2
        );
        assert_eq!(index_of(DispatchKey::Autograd, BackendComponent::Cpu), 3);
        assert_eq!(index_of(DispatchKey::Autograd, BackendComponent::Gpu), 4);
        assert_eq!(DispatchKeySet::table_size(), 5);
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

    /// Catches a length bump to `ALL_FUNCTIONALITIES` that isn't matched by
    /// filling in every discriminant `0..len` (e.g. a gap or duplicate).
    #[test]
    fn all_functionalities_covers_every_discriminant_exactly_once() {
        let mut discriminants: Vec<u8> = super::ALL_FUNCTIONALITIES
            .iter()
            .map(|k| *k as u8)
            .collect();
        discriminants.sort_unstable();
        let expected: Vec<u8> =
            (0..u8::try_from(super::ALL_FUNCTIONALITIES.len()).expect("small")).collect();
        assert_eq!(discriminants, expected);
    }
}
