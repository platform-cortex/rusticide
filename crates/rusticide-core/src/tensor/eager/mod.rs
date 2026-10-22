// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Legacy eager / SP0 port of the old eager `Tensor` API: data access
//! (`from_slice`, `get`, `set`, `to_vec`, `ndim`, `deep_copy`) as inherent
//! methods, and the legacy operators (`reshape`, `contiguous`, `add`,
//! `matmul`) behind [`EagerOps`].
//!
//! Legacy eager CPU API (SP0 port of the deleted rusticide-tensor). Bypasses
//! the dispatcher. Replaced by dispatched operators in SP2; do not add new
//! operators here.

mod access;
mod copy;
mod ops;

pub use ops::EagerOps;
