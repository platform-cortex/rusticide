// SPDX-FileCopyrightText: 2026 Rusticide contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rusticide core: value types and the dispatch key algebra.

mod device;
mod dtype;
mod error;
mod layout;
mod shape;
mod strides;

pub use device::{Device, DeviceKind};
pub use dtype::DType;
pub use error::{Error, Result};
pub use layout::{Layout, MemoryFormat};
pub use shape::Shape;
pub use strides::Strides;
