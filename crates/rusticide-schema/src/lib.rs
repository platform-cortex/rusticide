// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rusticide schema: a Rust port of torchgen's operator schema data model.

mod annotation;
mod argument;
mod base_ty;
mod error;
mod name;
mod returns;
mod ty;

pub use annotation::Annotation;
pub use argument::{Argument, Arguments};
pub use base_ty::BaseTy;
pub use error::ParseError;
pub use name::OperatorName;
pub use returns::Return;
pub use ty::Type;
