// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

/// Counts in-place mutations of a tensor's data.
///
/// This lives on the tensor itself rather than on autograd metadata. A
/// tensor that does not require grad has no autograd metadata at all, yet
/// it can still be saved for backward by an operation downstream of it;
/// lazily creating a counter at save time would race with a concurrent
/// forward pass mutating the same tensor. Putting the counter here instead
/// makes it unconditionally available, independent of whether autograd is
/// tracking this tensor.
///
/// Views and clones share a counter (it is itself reference-counted), so a
/// mutation through any of them is observed by all. Autograd compares a
/// saved tensor's recorded version against the live one to detect
/// "modified by an inplace operation" -- a runtime borrow check that turns
/// what would be a silently wrong gradient into a reported error.
#[derive(Debug, Clone, Default)]
pub struct VersionCounter(Arc<AtomicU32>);

impl VersionCounter {
    #[must_use]
    pub fn new() -> Self {
        Self(Arc::new(AtomicU32::new(0)))
    }

    #[must_use]
    pub fn get(&self) -> u32 {
        self.0.load(Ordering::Acquire)
    }

    pub fn bump(&self) {
        self.0.fetch_add(1, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::VersionCounter;

    #[test]
    fn a_new_counter_starts_at_zero() {
        assert_eq!(VersionCounter::new().get(), 0);
    }

    #[test]
    fn bump_increments_and_clones_share_the_count() {
        let a = VersionCounter::new();
        let b = a.clone();
        a.bump();
        assert_eq!(a.get(), 1);
        assert_eq!(b.get(), 1, "clones must share one counter");
    }
}
