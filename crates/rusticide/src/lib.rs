// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Rusticide -- a pure-Rust deep learning framework
//!
//! ```
//! use rusticide::{DType, Device, Tensor};
//!
//! let t = Tensor::zeros(&[2, 3], DType::F32, Device::cpu())?;
//! assert_eq!(t.shape().dims(), &[2, 3]);
//! # Ok::<(), rusticide::Error>(())
//! ```
//!
//! The legacy eager surface, `EagerOps`, is deliberately not re-exported;
//! import it from `rusticide_core` explicitly:
//!
//! ```compile_fail
//! use rusticide::EagerOps;
//! ```

pub use rusticide_core::{
    BackendComponent, DType, Device, DeviceKind, DispatchKey, DispatchKeySet, Element, Error,
    Layout, MemoryFormat, NUM_BACKENDS, Result, Shape, Storage, StorageReadGuard,
    StorageWriteGuard, Strides, Tensor, VersionCounter,
};

#[cfg(feature = "dispatch")]
pub use rusticide_dispatch as dispatch;
#[cfg(feature = "io")]
pub use rusticide_io as io;
#[cfg(feature = "models")]
pub use rusticide_model as model;
#[cfg(feature = "schema")]
pub use rusticide_schema as schema;

#[cfg(feature = "cli")]
pub mod cli;
