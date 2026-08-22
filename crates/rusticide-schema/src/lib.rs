// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rusticide schema: a Rust port of torchgen's operator schema data model.

mod error;
mod ty;

pub use error::ParseError;
pub use ty::{BaseTy, Type};
