// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

mod reshape;

use crate::{Result, Tensor};

/// Legacy eager CPU operators on [`Tensor`].
///
/// Legacy eager CPU operator (SP0 port of the deleted rusticide-tensor).
/// Bypasses the dispatcher. Replaced by dispatched operators in SP2; do not
/// add new operators here.
pub trait EagerOps: Sized {
    /// Legacy eager CPU operator (SP0 port of the deleted rusticide-tensor).
    /// Bypasses the dispatcher. Replaced by dispatched operators in SP2; do not
    /// add new operators here.
    ///
    /// Reshape to `dims`, row-major. Copies: the result owns new storage
    /// (tier-1 semantics; true views arrive with SP2).
    ///
    /// # Errors
    ///
    /// `Error::ElementCountMismatch` if `dims` holds a different element
    /// count; allocation errors from the copy.
    fn reshape(&self, dims: &[usize]) -> Result<Self>;

    /// Legacy eager CPU operator (SP0 port of the deleted rusticide-tensor).
    /// Bypasses the dispatcher. Replaced by dispatched operators in SP2; do not
    /// add new operators here.
    ///
    /// A contiguous tensor with the same data. Every tensor core can build
    /// is already contiguous with offset 0, so this returns an aliasing
    /// clone; SP2's views must make it copy when that stops holding.
    #[must_use]
    fn contiguous(&self) -> Self;
}

impl EagerOps for Tensor {
    fn reshape(&self, dims: &[usize]) -> Result<Self> {
        reshape::reshape(self, dims)
    }

    fn contiguous(&self) -> Self {
        reshape::contiguous(self)
    }
}
