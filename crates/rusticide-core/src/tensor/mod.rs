// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

mod eager;
mod handle;
mod impl_;
mod version;

pub use eager::EagerOps;
pub use handle::Tensor;
pub use version::VersionCounter;
