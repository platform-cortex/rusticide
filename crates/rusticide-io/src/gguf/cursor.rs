// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::{GgufError, Result};

/// A bounds-checked little-endian reader over untrusted bytes.
pub(crate) struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    pub(crate) const fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub(crate) const fn position(&self) -> usize {
        self.pos
    }

    pub(crate) const fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    pub(crate) fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or(GgufError::UnexpectedEof)?;
        let bytes = self
            .data
            .get(self.pos..end)
            .ok_or(GgufError::UnexpectedEof)?;
        self.pos = end;
        Ok(bytes)
    }

    pub(crate) fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut out = [0_u8; N];
        out.copy_from_slice(self.take(N)?);
        Ok(out)
    }

    pub(crate) fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    pub(crate) fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    /// Reads a `u64` length, saturating so the next bounds check rejects it.
    pub(crate) fn len(&mut self) -> Result<usize> {
        Ok(usize::try_from(self.u64()?).unwrap_or(usize::MAX))
    }

    pub(crate) fn string(&mut self) -> Result<String> {
        let len = self.len()?;
        Ok(String::from_utf8_lossy(self.take(len)?).into_owned())
    }
}

#[cfg(test)]
#[path = "cursor_tests.rs"]
mod tests;
