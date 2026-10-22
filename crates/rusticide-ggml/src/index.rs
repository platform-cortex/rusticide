// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_core::{Error, Shape};

use crate::Result;

pub(crate) fn unflatten_index(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut index = vec![0; shape.len()];
    for (slot, &extent) in index.iter_mut().zip(shape).rev() {
        *slot = flat % extent;
        flat /= extent;
    }
    index
}

pub(crate) fn flatten_index(index: &[usize], shape: &[usize]) -> usize {
    index
        .iter()
        .zip(shape)
        .fold(0, |flat, (&at, &extent)| flat * extent + at)
}

pub(crate) fn broadcast_map(result: &[usize], source: &[usize]) -> Vec<usize> {
    let offset = result.len() - source.len();
    source
        .iter()
        .zip(&result[offset..])
        .map(|(&extent, &at)| if extent == 1 { 0 } else { at })
        .collect()
}

pub(crate) fn ensure_repeatable(source: &[usize], target: &[usize]) -> Result<()> {
    let target_shape = Shape::new(target);
    if Shape::new(source).broadcast_with(&target_shape)? == target_shape {
        Ok(())
    } else {
        Err(Error::NotBroadcastable {
            lhs: source.to_vec(),
            rhs: target.to_vec(),
        }
        .into())
    }
}
