// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `BitNet` b1.58: configuration and ternary weights by tensor name, loaded
//! into a [`rusticide_transformer::Decoder`].

mod config;
mod model;

pub use config::{ARCHITECTURE, config};
pub use model::load;

#[cfg(test)]
mod testing;

#[cfg(test)]
#[path = "forward_tests.rs"]
mod forward_tests;
