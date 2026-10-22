// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::ops::{Deref, DerefMut};
use std::sync::{RwLockReadGuard, RwLockWriteGuard};

use rusticide_alloc::Buffer;

/// A shared view of a [`Storage`](super::Storage)'s bytes.
///
/// Derefs only to `[u8]`, never to the underlying buffer: the buffer's
/// length is fixed at `numel * dtype.size_in_bytes()` when the storage is
/// created, and handing out the buffer itself would let safe code replace
/// it and break that invariant.
#[derive(Debug)]
pub struct StorageReadGuard<'a>(RwLockReadGuard<'a, Buffer>);

impl<'a> StorageReadGuard<'a> {
    pub(super) const fn new(guard: RwLockReadGuard<'a, Buffer>) -> Self {
        Self(guard)
    }
}

impl Deref for StorageReadGuard<'_> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.0
    }
}

/// An exclusive view of a [`Storage`](super::Storage)'s bytes.
///
/// Derefs only to `[u8]`, so `guard.len() == numel * dtype.size_in_bytes()`
/// holds for as long as the storage lives; see [`StorageReadGuard`].
#[derive(Debug)]
pub struct StorageWriteGuard<'a>(RwLockWriteGuard<'a, Buffer>);

impl<'a> StorageWriteGuard<'a> {
    pub(super) const fn new(guard: RwLockWriteGuard<'a, Buffer>) -> Self {
        Self(guard)
    }
}

impl Deref for StorageWriteGuard<'_> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.0
    }
}

impl DerefMut for StorageWriteGuard<'_> {
    fn deref_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}
