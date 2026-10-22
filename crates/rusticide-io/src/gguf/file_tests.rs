// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::atomic::{AtomicUsize, Ordering};

use super::{GgufFile, align_up};
use crate::error::GgufError;
use crate::gguf::GGUFValue;
use crate::gguf::test_support::{GgufBuilder, header_bytes, tensor_info_bytes};

fn parse(bytes: Vec<u8>) -> crate::error::Result<GgufFile> {
    GgufFile::from_bytes(bytes)
}

#[test]
fn no_kv_pairs() {
    let file = parse(GgufBuilder::new().build()).unwrap();
    assert!(file.metadata().is_empty());
    assert_eq!(file.tensor_count(), 0);
}

#[test]
fn metadata_is_parsed_and_the_last_duplicate_key_wins() {
    let bytes = GgufBuilder::new()
        .kv("k", &GGUFValue::Uint32(1))
        .kv("k", &GGUFValue::Uint32(2))
        .build();
    let file = parse(bytes).unwrap();
    assert_eq!(file.metadata().get("k"), Some(&GGUFValue::Uint32(2)));
}

#[test]
fn an_invalid_value_type_in_metadata_is_rejected() {
    let bytes = GgufBuilder::new().raw_kv("k", 99, &[0; 4]).build();
    assert!(matches!(parse(bytes), Err(GgufError::InvalidValueType(99))));
}

#[test]
fn header_errors_surface_through_from_bytes() {
    assert!(matches!(
        parse(GgufBuilder::new().magic(*b"NOPE").build()),
        Err(GgufError::InvalidMagic)
    ));
    assert!(matches!(
        parse(GgufBuilder::new().version(2).build()),
        Err(GgufError::UnsupportedVersion(2))
    ));
}

#[test]
fn tensor_names_are_sorted() {
    let names = ["h", "d", "a", "g", "c", "f", "b", "e"];
    let builder = names.iter().fold(GgufBuilder::new(), |builder, name| {
        builder.tensor(name, &[1], 0, &[0; 4])
    });
    let file = parse(builder.build()).unwrap();
    assert_eq!(
        file.tensor_names(),
        ["a", "b", "c", "d", "e", "f", "g", "h"]
    );
}

#[test]
fn multiple_tensors_infos() {
    let bytes = GgufBuilder::new()
        .tensor("a", &[2, 3], 0, &[0; 24])
        .tensor("b", &[4], 1, &[0; 8])
        .build();
    let file = parse(bytes).unwrap();
    assert_eq!(file.tensor_count(), 2);
    assert_eq!(file.tensor_info("a").unwrap().shape, [3, 2]);
    assert_eq!(file.tensor_info("a").unwrap().offset, 0);
    assert_eq!(file.tensor_info("b").unwrap().offset, 32);
}

#[test]
fn tensor_info_not_found() {
    assert!(
        parse(GgufBuilder::new().build())
            .unwrap()
            .tensor_info("x")
            .is_none()
    );
}

#[test]
fn duplicate_tensor_names_are_rejected() {
    let bytes = GgufBuilder::new()
        .tensor("w", &[1], 0, &[0; 4])
        .tensor("w", &[1], 0, &[0; 4])
        .build();
    assert!(matches!(
        parse(bytes),
        Err(GgufError::DuplicateTensor { name }) if name == "w"
    ));
}

#[test]
fn tensor_count_and_names_always_agree() {
    let bytes = GgufBuilder::new().tensor("a", &[1], 0, &[0; 4]).build();
    let file = parse(bytes).unwrap();
    assert_eq!(file.tensor_count(), file.tensor_names().len());
}

#[test]
fn tensor_count_bound_rejects_a_tiny_file() {
    let bytes = header_bytes(3, u64::MAX, 0);
    assert!(matches!(parse(bytes), Err(GgufError::UnexpectedEof)));
}

#[test]
fn kv_count_bound_rejects_a_tiny_file() {
    let bytes = header_bytes(3, 0, u64::MAX);
    assert!(matches!(parse(bytes), Err(GgufError::UnexpectedEof)));
}

#[test]
fn a_tensor_info_error_surfaces() {
    let mut bytes = header_bytes(3, 1, 0);
    bytes.extend(tensor_info_bytes("t", &[1; 5], 0, 0));
    assert!(matches!(
        parse(bytes),
        Err(GgufError::TooManyDimensions { got: 5, max: 4 })
    ));
}

#[test]
fn data_start_is_aligned_to_the_default() {
    let bytes = GgufBuilder::new().tensor("a", &[1], 0, &[7; 4]).build();
    let file = parse(bytes).unwrap();
    assert_eq!(file.data_start % 32, 0);
    assert_eq!(file.data[file.data_start], 7);
}

#[test]
fn data_start_honours_general_alignment() {
    let bytes = GgufBuilder::new()
        .alignment(64)
        .tensor("a", &[1], 0, &[7; 4])
        .build();
    let file = parse(bytes).unwrap();
    assert_eq!(file.data_start % 64, 0);
    assert_eq!(file.data[file.data_start], 7);
}

#[test]
fn invalid_alignment_rejects_the_file() {
    for (value, bad) in [
        (GGUFValue::Uint32(0), "zero"),
        (GGUFValue::Uint32(48), "non power of two"),
        (GGUFValue::String("32".into()), "wrong type"),
    ] {
        let bytes = GgufBuilder::new().kv("general.alignment", &value).build();
        assert!(
            matches!(parse(bytes), Err(GgufError::InvalidAlignment(found)) if found == value),
            "{bad}"
        );
    }
}

#[test]
fn align_up_rounds_to_a_multiple() {
    assert_eq!(align_up(0, 32).unwrap(), 0);
    assert_eq!(align_up(1, 32).unwrap(), 32);
    assert_eq!(align_up(32, 32).unwrap(), 32);
    assert_eq!(align_up(33, 64).unwrap(), 64);
}

#[test]
fn align_up_overflow_is_cast_overflow() {
    assert!(matches!(
        align_up(usize::MAX, 32),
        Err(GgufError::CastOverflow("tensor data start"))
    ));
}

#[test]
fn load_reads_a_file_from_disk() {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "rusticide-io-{}-{}.gguf",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(
        &path,
        GgufBuilder::new().tensor("a", &[1], 0, &[0; 4]).build(),
    )
    .unwrap();
    let loaded = GgufFile::load(&path);
    std::fs::remove_file(&path).unwrap();
    assert_eq!(loaded.unwrap().tensor_names(), ["a"]);
}

#[test]
fn load_of_a_missing_file_is_io() {
    let path = std::env::temp_dir().join("rusticide-io-definitely-missing.gguf");
    assert!(matches!(GgufFile::load(path), Err(GgufError::Io(_))));
}

#[test]
fn a_data_section_starting_past_eof_is_rejected() {
    let mut bytes = GgufBuilder::new().build();
    bytes.truncate(bytes.len() - 8);
    assert!(matches!(parse(bytes), Err(GgufError::UnexpectedEof)));
}

#[test]
fn a_data_section_starting_exactly_at_eof_is_accepted() {
    let file = parse(GgufBuilder::new().build()).unwrap();
    assert_eq!(file.data_start, file.data.len());
}

#[test]
fn a_misaligned_tensor_offset_is_rejected() {
    let mut bytes = header_bytes(3, 1, 0);
    bytes.extend(tensor_info_bytes("w", &[1], 0, 4));
    bytes.resize(bytes.len().next_multiple_of(32) + 32, 0);
    assert!(matches!(
        parse(bytes),
        Err(GgufError::MisalignedTensor { name, offset: 4, alignment: 32 }) if name == "w"
    ));
}

#[test]
fn a_tensor_offset_aligned_to_a_custom_alignment_is_accepted() {
    let bytes = GgufBuilder::new()
        .alignment(64)
        .tensor("a", &[1], 0, &[0; 4])
        .tensor("b", &[1], 0, &[0; 4])
        .build();
    assert_eq!(parse(bytes).unwrap().tensor_info("b").unwrap().offset, 64);
}
