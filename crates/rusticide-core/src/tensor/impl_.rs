// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{DispatchKeySet, Shape, Storage, Strides};

use super::version::VersionCounter;

/// The data behind a `Tensor` handle.
///
/// `Tensor` wraps this in an `Arc` so that cloning a tensor aliases rather
/// than copies: clones share the same `storage` (and, through it, the same
/// underlying buffer) and the same `version` counter. This is required, not
/// incidental -- a training loop mutates parameters that a module and an
/// optimizer both hold, and both must observe the update.
#[derive(Debug)]
pub(crate) struct TensorImpl {
    pub(crate) storage: Storage,
    pub(crate) shape: Shape,
    pub(crate) strides: Strides,
    pub(crate) storage_offset: usize,
    pub(crate) version: VersionCounter,
    pub(crate) key_set: DispatchKeySet,
}
