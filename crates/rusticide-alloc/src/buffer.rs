// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::ops::{Deref, DerefMut};

/// An owned, byte-addressed allocation backing a tensor's storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Buffer {
    bytes: Vec<u8>,
}

impl Buffer {
    /// Allocate `bytes` zero-filled bytes.
    ///
    /// Returns `None` if the allocation fails.
    #[must_use]
    pub fn alloc_zeroed(bytes: usize) -> Option<Self> {
        let mut v = Vec::new();
        v.try_reserve_exact(bytes).ok()?;
        v.resize(bytes, 0);
        Some(Self { bytes: v })
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.bytes.len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.bytes
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.bytes
    }
}

impl Deref for Buffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.bytes
    }
}

impl DerefMut for Buffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::Buffer;

    #[test]
    fn zeroed_buffer_has_the_requested_length_and_is_zero() {
        let b = Buffer::alloc_zeroed(64).expect("allocation succeeds");
        assert_eq!(b.len(), 64);
        assert!(b.as_slice().iter().all(|&x| x == 0));
    }

    #[test]
    fn a_zero_length_buffer_is_valid_and_empty() {
        let b = Buffer::alloc_zeroed(0).expect("allocation succeeds");
        assert_eq!(b.len(), 0);
        assert_eq!(b.as_slice(), []);
    }

    #[test]
    fn writes_through_the_mutable_slice_are_visible() {
        let mut b = Buffer::alloc_zeroed(4).expect("allocation succeeds");
        b.as_mut_slice()[2] = 7;
        assert_eq!(b.as_slice(), &[0, 0, 7, 0]);
    }

    #[test]
    fn indexing_through_deref_reads_what_was_written() {
        let mut b = Buffer::alloc_zeroed(4).expect("allocation succeeds");
        b.as_mut_slice()[1] = 9;
        assert_eq!(b[1], 9);
    }

    #[test]
    fn deref_len_matches_buffer_len() {
        let b = Buffer::alloc_zeroed(10).expect("allocation succeeds");
        assert_eq!(b.len(), b.as_slice().len());
        let deref_len: usize = (*b).len();
        assert_eq!(deref_len, b.len());
    }

    #[test]
    fn a_mutation_through_deref_mut_is_visible_through_a_later_deref() {
        let mut b = Buffer::alloc_zeroed(4).expect("allocation succeeds");
        b[3] = 42;
        assert_eq!(&*b, &[0, 0, 0, 42]);
    }
}
