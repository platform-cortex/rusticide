// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rusticide dispatch: the boxed calling convention and operator dispatcher.

pub mod aliasing;
mod dispatcher;
mod entry;
mod error;
mod ivalue;
mod kernel;
mod stack;

pub use dispatcher::Dispatcher;
pub use entry::OperatorEntry;
pub use error::DispatchError;
pub use ivalue::IValue;
pub use kernel::KernelFunction;
pub use stack::Stack;
