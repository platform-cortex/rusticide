// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Maps a [`DispatchKeySet`] to a slot in an operator's kernel table.
//!
//! Kept separate from `key_set`'s bit-packing algebra: this module encodes a
//! layout *policy* (how many slots each functionality claims) on top of that
//! algebra, not the algebra itself.

use super::DispatchKeySet;
use super::key_set::ALL_FUNCTIONALITIES;

/// Kernel-table slots a per-backend functionality occupies: one per backend.
const BACKENDS_PER_FUNCTIONALITY: usize = 2;

impl DispatchKeySet {
    /// The slot in an operator's kernel table this keyset resolves to, or
    /// `None` if this keyset carries no functionality at all.
    ///
    /// `DispatchKey` has no reserved "undefined" discriminant -- `Dense == 0`
    /// is a real, dispatchable slot, so an empty keyset cannot be told apart
    /// from `(Dense, Cpu)` by slot number alone. Returning `Option` instead of
    /// defaulting an empty keyset to slot 0 makes that distinction explicit
    /// and forces the caller to handle "no key" rather than silently routing
    /// to the Dense-CPU kernel.
    ///
    /// Per-backend functionalities occupy `BACKENDS_PER_FUNCTIONALITY`
    /// consecutive slots (one per backend); others occupy one. Functionalities
    /// are laid out in discriminant order, so a functionality's base offset is
    /// the summed slot count of every functionality that outranks it less.
    #[must_use]
    pub fn table_index(self) -> Option<usize> {
        let f = self.highest_functionality()?;
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
        Some(if f.is_per_backend() {
            offset + self.highest_backend().map_or(0, |b| b as usize - 1)
        } else {
            offset
        })
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
}

#[cfg(test)]
mod tests {
    use super::DispatchKeySet;
    use crate::{BackendComponent, DispatchKey};

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
        assert_eq!(index_of(DispatchKey::Dense, BackendComponent::Cpu), Some(0));
        assert_eq!(index_of(DispatchKey::Dense, BackendComponent::Gpu), Some(1));
        assert_eq!(
            index_of(DispatchKey::Functionalize, BackendComponent::Cpu),
            Some(2)
        );
        assert_eq!(
            index_of(DispatchKey::Functionalize, BackendComponent::Gpu),
            Some(2)
        );
        assert_eq!(
            index_of(DispatchKey::Autograd, BackendComponent::Cpu),
            Some(3)
        );
        assert_eq!(
            index_of(DispatchKey::Autograd, BackendComponent::Gpu),
            Some(4)
        );
        assert_eq!(DispatchKeySet::table_size(), 5);
    }

    /// `Dense == 0` is a real slot, not a sentinel, so `table_index` must be
    /// able to tell an empty keyset apart from `(Dense, Cpu)` -- the whole
    /// reason it returns `Option<usize>` instead of defaulting to slot 0.
    #[test]
    fn an_empty_keyset_does_not_alias_the_dense_cpu_slot() {
        let empty = DispatchKeySet::empty();
        let dense_cpu = DispatchKeySet::empty()
            .with_backend(BackendComponent::Cpu)
            .with_functionality(DispatchKey::Dense);
        assert_eq!(empty.table_index(), None);
        assert_eq!(dense_cpu.table_index(), Some(0));
    }
}
