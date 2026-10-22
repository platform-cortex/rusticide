// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

mod binary;
mod rows;
mod structural;

use rusticide_core::{EagerOps, Tensor};

use crate::{Op, Result};

/// Run `op` over `inputs`, whose count the builder has already checked.
pub(crate) fn execute_op(op: &Op, inputs: &[&Tensor]) -> Result<Tensor> {
    match op {
        Op::Copy => structural::copy(inputs[0]),
        Op::Reshape { shape } => structural::reshape(inputs[0], shape),
        Op::Permute | Op::Cont => Ok(inputs[0].contiguous()),
        Op::Repeat { shape } => structural::repeat(inputs[0], shape),
        Op::GetRows => rows::get_rows(inputs[0], inputs[1]),
        Op::Add => binary::add(inputs[0], inputs[1]),
        Op::MatMul => binary::matmul(inputs[0], inputs[1]),
    }
}

#[cfg(test)]
mod tests;
