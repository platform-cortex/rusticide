// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Family-independent model runtime: sources, weights and the KV cache.

mod cache;
mod error;
mod model;
mod registry;
mod source;
mod weights;

pub use cache::KvCache;
pub use error::{ModelError, Result};
pub use model::Model;
pub use registry::{Loader, Registry};
pub use source::{InMemorySource, ModelSource};
pub use weights::{Matrix, load_vector};
