// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

mod dispatch_key;
mod kernel_table;
mod key_set;

pub use dispatch_key::{BackendComponent, DispatchKey, NUM_BACKENDS};
pub use key_set::DispatchKeySet;
