// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fuzz every block dequantizer against arbitrary block bytes.
//!
//! The first byte picks a registered format, the rest is block data sized to
//! whole blocks plus the format's trailer (the `I2_S` scale), so a hostile
//! scale reaches it too. A dequantizer must never panic or read past a block,
//! whole tensor or single row.

#![no_main]

use libfuzzer_sys::fuzz_target;
use rusticide_quantize::formats;

fuzz_target!(|data: &[u8]| {
    let Some((&selector, payload)) = data.split_first() else {
        return;
    };
    let table = formats();
    let format = &table[usize::from(selector) % table.len()];
    let blocks = payload.len().saturating_sub(format.trailer_bytes()) / format.block_bytes();
    let numel = blocks * format.block_size();
    let _ = format.dequantize(payload, &[numel]);
    let mut row = vec![0.0; numel];
    format.dequantize_row(payload, &mut row);
});
