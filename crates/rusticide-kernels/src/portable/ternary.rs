// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `I2_S` ternary rows against [`Q8Kind::Row`](crate::Q8Kind::Row)
//! activations, in bitnet.cpp's legacy arithmetic.

use crate::q8_input::Q8Input;

/// Bytes per `I2_S` block of 128 weights.
const BLOCK_BYTES: usize = 32;

/// Activation blocks per weight block: byte `k` holds weights `k`, `k + 32`,
/// `k + 64` and `k + 96`, at these shifts.
const LANES: usize = 4;
const SHIFTS: [u8; LANES] = [6, 4, 2, 0];

/// `sum((code - 1) * q)` over the whole blocks both operands hold, as
/// `sum(code * q) - sum(q)`. Exact below 5.5M columns, where the sum would
/// leave `i32`; a hostile width wraps instead of panicking.
pub(crate) fn ternary_dot(row: &[u8], input: &Q8Input) -> i32 {
    let (blocks, _) = row.as_chunks::<BLOCK_BYTES>();
    blocks
        .iter()
        .zip(input.spans::<LANES>())
        .map(|(block, span)| {
            let products: i32 = span
                .quants
                .iter()
                .zip(SHIFTS)
                .map(|(quants, shift)| {
                    block
                        .iter()
                        .zip(quants)
                        .map(|(&byte, &q)| i32::from((byte >> shift) & 0b11) * i32::from(q))
                        .sum::<i32>()
                })
                .sum();
            products - span.sums.iter().sum::<i32>()
        })
        .fold(0, i32::wrapping_add)
}

/// One row against one token: `dot / s * scale`, the legacy order. The C
/// computes `((float) sum(code * q) - (float) sum(q)) / s * scale`; both
/// sums stay below 2^24 for rows under 43690 columns, so converting their
/// exact difference once gives the same bits.
fn product(row: &[u8], input: &Q8Input, scale: f32) -> f32 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "dots of real widths are below 2^24 and convert exactly"
    )]
    let dot = ternary_dot(row, input) as f32;
    input
        .scales
        .first()
        .map_or(0.0, |&divisor| dot / divisor * scale)
}

/// The products of every `row_bytes` row of `data` with each input, on the
/// calling thread: `out[row * inputs.len() + token]`.
///
/// Each input is quantized as [`Q8Kind::Row`](crate::Q8Kind::Row) and
/// `scale` is the tensor's weight scale. `out.len()` bounds the rows
/// computed; nothing is written when `row_bytes` is zero.
pub fn dot_ternary_matrix(
    data: &[u8],
    row_bytes: usize,
    scale: f32,
    inputs: &[Q8Input],
    out: &mut [f32],
) {
    if row_bytes == 0 {
        return;
    }
    let tokens = inputs.len().max(1);
    for (slots, row) in out.chunks_mut(tokens).zip(data.chunks_exact(row_bytes)) {
        for (slot, input) in slots.iter_mut().zip(inputs) {
            *slot = product(row, input, scale);
        }
    }
}

#[cfg(test)]
#[path = "ternary_tests.rs"]
mod tests;
