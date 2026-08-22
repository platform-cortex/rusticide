// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rusticide schema: a Rust port of torchgen's operator schema data model.

mod annotation;
mod base_ty;
mod error;
mod ty;

pub use annotation::Annotation;
pub use base_ty::BaseTy;
pub use error::ParseError;
pub use ty::Type;
