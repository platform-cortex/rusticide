// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::test_support::{PAIRED, PAIRED_Q8_0};

static NEON_NAMED: Adapter = Adapter {
    name: "neon",
    ..portable::ADAPTER
};

static AVX2_NAMED: Adapter = Adapter {
    name: "avx2",
    ..portable::ADAPTER
};

fn name_of(chosen: (&'static Adapter, Option<String>)) -> (&'static str, Option<String>) {
    (chosen.0.name(), chosen.1)
}

fn names(adapters: &[&'static Adapter]) -> Vec<&'static str> {
    adapters.iter().map(|adapter| adapter.name()).collect()
}

const BOTH: [&Adapter; 2] = [&NEON_NAMED, &portable::ADAPTER];
const PORTABLE: [&Adapter; 1] = [&portable::ADAPTER];

#[test]
fn available_ends_with_portable() {
    let available = available();
    assert_eq!(
        available.last().map(|adapter| adapter.name()),
        Some("portable")
    );
    #[cfg(target_arch = "aarch64")]
    let simd = Some("neon");
    #[cfg(target_arch = "x86_64")]
    let simd = std::arch::is_x86_feature_detected!("avx2").then_some("avx2");
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let simd = None;
    let expected: Vec<_> = simd.into_iter().chain(["portable"]).collect();
    assert_eq!(names(&available), expected);
}

#[test]
fn ranked_puts_each_detected_adapter_before_portable() {
    assert_eq!(names(&ranked(&[Some(&NEON_NAMED)])), ["neon", "portable"]);
    assert_eq!(names(&ranked(&[None])), ["portable"]);
    assert_eq!(
        names(&ranked(&[None, Some(&AVX2_NAMED)])),
        ["avx2", "portable"]
    );
}

#[test]
fn active_adapter_is_the_best_available() {
    let best = available()[0].name();
    let overridden = std::env::var("RUSTICIDE_KERNELS").is_ok_and(|value| value == "portable");
    assert_eq!(active().name(), if overridden { "portable" } else { best });
    assert!(std::ptr::eq(active(), active()));
}

#[test]
fn unset_picks_the_best_available() {
    assert_eq!(name_of(choose(None, &BOTH)), ("neon", None));
    assert_eq!(name_of(choose(None, &PORTABLE)), ("portable", None));
}

#[test]
fn an_available_name_is_honoured() {
    assert_eq!(name_of(choose(Some("neon"), &BOTH)), ("neon", None));
    assert_eq!(name_of(choose(Some("portable"), &BOTH)), ("portable", None));
}

#[test]
fn portable_is_honoured_even_when_not_listed() {
    assert_eq!(
        name_of(choose(Some("portable"), &[&NEON_NAMED])),
        ("portable", None)
    );
}

#[test]
fn a_known_adapter_this_cpu_lacks_warns_and_uses_the_best() {
    assert_eq!(
        name_of(choose(Some("neon"), &PORTABLE)),
        (
            "portable",
            Some("RUSTICIDE_KERNELS=neon: not supported by this CPU, using portable".to_owned())
        )
    );
    assert_eq!(
        name_of(choose(Some("avx2"), &BOTH)),
        (
            "neon",
            Some("RUSTICIDE_KERNELS=avx2: not supported by this CPU, using neon".to_owned())
        )
    );
}

#[test]
fn unknown_value_warns_and_uses_the_best() {
    assert_eq!(
        name_of(choose(Some("avx"), &BOTH)),
        (
            "neon",
            Some("RUSTICIDE_KERNELS=avx: unknown adapter, using neon".to_owned())
        )
    );
    assert_eq!(
        name_of(choose(Some(""), &PORTABLE)),
        (
            "portable",
            Some("RUSTICIDE_KERNELS=: unknown adapter, using portable".to_owned())
        )
    );
}

#[test]
fn an_empty_list_falls_back_to_portable() {
    assert_eq!(name_of(choose(None, &[])), ("portable", None));
}

#[test]
fn select_writes_the_warning_as_one_line() {
    let mut sink = Vec::new();
    let adapter = select(Some("avx"), &BOTH, &mut sink);
    assert_eq!(adapter.name(), "neon");
    assert_eq!(
        String::from_utf8(sink).unwrap(),
        "RUSTICIDE_KERNELS=avx: unknown adapter, using neon\n"
    );
}

#[test]
fn select_is_silent_without_a_warning() {
    let mut sink = Vec::new();
    select(None, &BOTH, &mut sink);
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
    for adapter in available().into_iter().chain([&PAIRED]) {
        for layout in adapter.quantized.iter().filter_map(|kernel| kernel.layout) {
            assert!(layout.group_rows > 1, "{}", adapter.name());
        }
    }
    assert_eq!(PAIRED_Q8_0.layout.map(|layout| layout.group_rows), Some(2));
}
