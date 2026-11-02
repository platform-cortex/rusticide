// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::adapter::Adapter;
use crate::portable;

/// Starts as the portable kernels; NEON kernels replace them one by one.
pub(crate) static ADAPTER: Adapter = Adapter {
    name: "neon",
    ..portable::ADAPTER
};
