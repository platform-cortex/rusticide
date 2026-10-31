// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fuzz every block dequantizer against arbitrary block bytes.
//!
//! The first byte picks a registered format, the rest is block data sized to
//! whole blocks. A dequantizer must never panic or read past a block.

#![no_main]

use libfuzzer_sys::fuzz_target;
use rusticide_quantize::formats;

fuzz_target!(|data: &[u8]| {
    let Some((&selector, payload)) = data.split_first() else {
        return;
    };
    let table = formats();
    let format = &table[usize::from(selector) % table.len()];
    let blocks = payload.len() / format.block_bytes();
    let _ = format.dequantize(payload, &[blocks * format.block_size()]);
});
