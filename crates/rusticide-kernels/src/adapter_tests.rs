// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::test_support::{PAIRED, PAIRED_Q8_0};

static NEON_NAMED: Adapter = Adapter {
    name: "neon",
    ..portable::ADAPTER
};

fn name_of(chosen: (&'static Adapter, Option<String>)) -> (&'static str, Option<String>) {
    (chosen.0.name(), chosen.1)
}

#[test]
fn active_adapter_is_the_compiled_default() {
    let expected = if cfg!(all(target_arch = "aarch64", target_feature = "dotprod")) {
        "neon"
    } else {
        "portable"
    };
    assert_eq!(compiled_default().name(), expected);
    let overridden = std::env::var("RUSTICIDE_KERNELS").is_ok_and(|value| value == "portable");
    assert_eq!(
        active().name(),
        if overridden { "portable" } else { expected }
    );
    assert!(std::ptr::eq(active(), active()));
}

#[test]
fn unset_picks_the_compiled_default() {
    assert_eq!(name_of(choose(None, &NEON_NAMED)), ("neon", None));
    assert_eq!(
        name_of(choose(None, &portable::ADAPTER)),
        ("portable", None)
    );
}

#[test]
fn portable_is_always_honoured() {
    assert_eq!(
        name_of(choose(Some("portable"), &NEON_NAMED)),
        ("portable", None)
    );
}

#[test]
fn neon_is_honoured_when_compiled() {
    assert_eq!(name_of(choose(Some("neon"), &NEON_NAMED)), ("neon", None));
}

#[test]
fn neon_without_the_adapter_warns_and_uses_portable() {
    assert_eq!(
        name_of(choose(Some("neon"), &portable::ADAPTER)),
        (
            "portable",
            Some("RUSTICIDE_KERNELS=neon: adapter not compiled in, using portable".to_owned())
        )
    );
}

#[test]
fn unknown_value_warns_and_uses_the_compiled_default() {
    assert_eq!(
        name_of(choose(Some("avx"), &NEON_NAMED)),
        (
            "neon",
            Some("RUSTICIDE_KERNELS=avx: unknown adapter, using neon".to_owned())
        )
    );
    assert_eq!(
        name_of(choose(Some(""), &portable::ADAPTER)),
        (
            "portable",
            Some("RUSTICIDE_KERNELS=: unknown adapter, using portable".to_owned())
        )
    );
}

#[test]
fn select_writes_the_warning_as_one_line() {
    let mut sink = Vec::new();
    let adapter = select(Some("avx"), &NEON_NAMED, &mut sink);
    assert_eq!(adapter.name(), "neon");
    assert_eq!(
        String::from_utf8(sink).unwrap(),
        "RUSTICIDE_KERNELS=avx: unknown adapter, using neon\n"
    );
}

#[test]
fn select_is_silent_without_a_warning() {
    let mut sink = Vec::new();
    select(None, &NEON_NAMED, &mut sink);
    assert_eq!(sink, Vec::<u8>::new());
}

fn q8_0_rows(rows: usize) -> (Vec<u8>, usize) {
    let format = rusticide_quantize::block_format(8).unwrap();
    let data = format.synthetic_blocks(rows * 2, 3).unwrap();
    let row_bytes = format.row_bytes(64).unwrap();
    (data, row_bytes)
}

#[test]
fn prepare_matrix_falls_back_on_an_odd_row_count() {
    let (original, row_bytes) = q8_0_rows(3);
    let mut data = original.clone();
    let kernel = PAIRED.prepare_matrix(8, &mut data, row_bytes).unwrap();
    assert!(std::ptr::eq(
        kernel,
        portable::ADAPTER.quant_kernel(8).unwrap()
    ));
    assert_eq!(data, original);
}

#[test]
fn prepare_matrix_repacks_whole_groups_in_place() {
    let (original, row_bytes) = q8_0_rows(4);
    let mut data = original.clone();
    let kernel = PAIRED.prepare_matrix(8, &mut data, row_bytes).unwrap();
    assert!(std::ptr::eq(kernel, std::ptr::from_ref(&PAIRED_Q8_0)));
    assert_eq!(data[..row_bytes], original[row_bytes..2 * row_bytes]);
}

#[test]
fn prepare_matrix_without_a_kernel_uses_portable_or_none() {
    let (original, row_bytes) = q8_0_rows(4);
    let mut data = original.clone();
    let portable_q4_k = portable::ADAPTER.quant_kernel(12).unwrap();
    let kernel = PAIRED.prepare_matrix(12, &mut data, row_bytes).unwrap();
    assert!(std::ptr::eq(kernel, portable_q4_k));
    assert!(PAIRED.prepare_matrix(99, &mut data, row_bytes).is_none());
    let identity = portable::ADAPTER.prepare_matrix(8, &mut data, row_bytes);
    assert!(std::ptr::eq(
        identity.unwrap(),
        portable::ADAPTER.quant_kernel(8).unwrap()
    ));
    assert_eq!(data, original);
}

#[test]
fn every_layout_groups_more_than_one_row() {
    for adapter in [&portable::ADAPTER, compiled_default(), &PAIRED] {
        for layout in adapter.quantized.iter().filter_map(|kernel| kernel.layout) {
            assert!(layout.group_rows > 1, "{}", adapter.name());
        }
    }
    assert_eq!(PAIRED_Q8_0.layout.map(|layout| layout.group_rows), Some(2));
}
