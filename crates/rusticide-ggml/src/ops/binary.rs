// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_core::{EagerOps, Tensor};

use crate::Result;

pub(super) fn add(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    Ok(lhs.add(rhs)?)
}

pub(super) fn matmul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    Ok(lhs.matmul(rhs)?)
}
