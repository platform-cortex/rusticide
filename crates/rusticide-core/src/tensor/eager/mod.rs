// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Legacy eager / SP0 port: data access for the old eager `Tensor` API.
//!
//! Legacy eager CPU API (SP0 port of the deleted rusticide-tensor). Bypasses
//! the dispatcher. Replaced by dispatched operators in SP2; do not add new
//! operators here.

mod access;
