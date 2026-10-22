// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_core::Tensor;

use crate::Result;

pub(super) fn scale(input: &Tensor, factor: f32) -> Result<Tensor> {
    map_f32(input, |value| value * factor)
}

pub(super) fn silu(input: &Tensor) -> Result<Tensor> {
    map_f32(input, |value| value / (1.0 + (-value).exp()))
}

/// Applies `function` to every element, in place on the fallibly allocated
/// copy `to_vec` returns.
fn map_f32(input: &Tensor, function: impl Fn(f32) -> f32) -> Result<Tensor> {
    let mut values = input.to_vec::<f32>()?;
    for value in &mut values {
        *value = function(*value);
    }
    Ok(Tensor::from_slice(&values, input.shape().dims())?)
}

#[cfg(test)]
mod tests;
