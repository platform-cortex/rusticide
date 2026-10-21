// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rusticide alloc: aligned buffer allocation for tensor storage.
//!
//! This is the first crate in the workspace permitted to use `unsafe`. See
//! `docs/UNSAFE.md` for the policy every `unsafe`-permitted crate must
//! satisfy, and this crate's `README.md` for why safe Rust alone will not
//! remain sufficient.

mod buffer;

pub use buffer::Buffer;
