// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rusticide core: value types and the dispatch key algebra.

mod device;
mod dtype;
mod element;
mod error;
mod key;
mod layout;
mod reserve;
mod shape;
mod storage;
mod strides;
mod tensor;

pub use device::{Device, DeviceKind};
pub use dtype::DType;
pub use element::Element;
pub use error::{Error, Result};
pub use key::{BackendComponent, DispatchKey, DispatchKeySet, NUM_BACKENDS};
pub use layout::{Layout, MemoryFormat};
pub use reserve::try_vec_with_capacity;
pub use shape::Shape;
pub use storage::{Storage, StorageReadGuard, StorageWriteGuard};
pub use strides::Strides;
pub use tensor::{EagerOps, Tensor, VersionCounter};
