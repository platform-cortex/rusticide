// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rusticide dispatch: the boxed calling convention and operator dispatcher.

mod ivalue;
mod stack;

pub use ivalue::IValue;
pub use stack::Stack;
