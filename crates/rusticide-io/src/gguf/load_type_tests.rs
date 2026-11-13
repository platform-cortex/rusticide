// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use half::f16;
use rusticide_quantize::QuantizeError;

use super::GgufFile;
use crate::error::GgufError;
use crate::gguf::GGUFValue;
use crate::gguf::test_support::GgufBuilder;

const F32: u32 = 0;

fn file(builder: GgufBuilder) -> GgufFile {
    GgufFile::from_bytes(builder.build()).unwrap()
}

fn one_f32() -> Vec<u8> {
    1.0_f32.to_le_bytes().to_vec()
}

#[test]
fn unknown_type_fails_only_when_materialized() {
    let file = file(
        GgufBuilder::new()
            .kv(
                "general.architecture",
                &GGUFValue::String("llama".to_owned()),
            )
            .tensor("good", &[1], F32, &one_f32())
            .tensor("odd", &[1], 99, &[0; 4]),
    );
    assert_eq!(file.metadata().architecture(), Some("llama"));
    assert_eq!(file.tensor("good").unwrap().to_vec::<f32>().unwrap(), [1.0]);
    assert!(matches!(
        file.tensor("odd"),
        Err(GgufError::UnsupportedTensorType { name, type_id: 99 }) if name == "odd"
    ));
}

#[test]
fn unsupported_type_message_names_the_tensor() {
    let file = file(GgufBuilder::new().tensor("odd", &[1], 99, &[0; 4]));
    let error = file.tensor("odd").unwrap_err();
    assert_eq!(error.to_string(), "tensor odd has unsupported type 99");
}

#[test]
fn q8_0_tensor_loads_through_the_table() {
    let mut block = f16::from_f32(0.5).to_le_bytes().to_vec();
    block.extend([0x80, 0xFF, 0, 1, 127]);
    block.extend([0; 27]);
    let file = file(GgufBuilder::new().tensor("q", &[32], 8, &block));
    let values = file.tensor("q").unwrap().to_vec::<f32>().unwrap();
    assert_eq!(values[..6], [-64.0, -0.5, 0.0, 0.5, 63.5, 0.0]);
}

#[test]
fn quantized_tensor_borrows_the_raw_blocks() {
    let mut block = f16::from_f32(0.5).to_le_bytes().to_vec();
    block.extend([7; 32]);
    let file = file(GgufBuilder::new().tensor("q", &[32], 8, &block).tensor(
        "f",
        &[1],
        F32,
        &one_f32(),
    ));
    let (format, bytes) = file.quantized_tensor("q").unwrap().unwrap();
    assert_eq!((format.name(), bytes), ("Q8_0", &block[..]));
    assert!(file.quantized_tensor("f").unwrap().is_none());
    assert!(matches!(
        file.quantized_tensor("nope"),
        Err(GgufError::TensorNotFound(name)) if name == "nope"
    ));
}

#[test]
fn quantized_tensor_of_an_unknown_type_is_unsupported() {
    let file = file(GgufBuilder::new().tensor("odd", &[1], 99, &[0; 4]));
    assert!(matches!(
        file.quantized_tensor("odd"),
        Err(GgufError::UnsupportedTensorType { type_id: 99, .. })
    ));
}

const I2_S: u32 = 36;

/// One `I2_S` block of +1 codes, then the trailer: scale 0.5 and garbage.
fn i2_s_tensor() -> Vec<u8> {
    let mut data = vec![0b1010_1010; 32];
    data.extend(0.5_f32.to_le_bytes());
    data.extend([0xA5; 28]);
    data
}

#[test]
fn an_i2_s_tensor_loads_scaled_and_borrows_its_trailer() {
    let file = file(GgufBuilder::new().tensor("t", &[128], I2_S, &i2_s_tensor()));
    let values = file.tensor("t").unwrap().to_vec::<f32>().unwrap();
    assert_eq!(values, [0.5; 128]);
    let (format, bytes) = file.quantized_tensor("t").unwrap().unwrap();
    assert_eq!((format.name(), bytes), ("I2_S", &i2_s_tensor()[..]));
}

#[test]
fn an_i2_s_tensor_without_its_trailer_is_truncated() {
    let file = file(GgufBuilder::new().tensor("t", &[128], I2_S, &[0; 32]));
    assert!(matches!(file.tensor("t"), Err(GgufError::UnexpectedEof)));
    assert!(matches!(
        file.quantized_tensor("t"),
        Err(GgufError::UnexpectedEof)
    ));
}

#[test]
fn an_i2_s_tensor_that_splits_a_block_is_unaligned() {
    let file = file(GgufBuilder::new().tensor("t", &[100], I2_S, &i2_s_tensor()));
    assert!(matches!(
        file.tensor("t"),
        Err(GgufError::Quantize(QuantizeError::UnalignedShape {
            format: "I2_S",
            numel: 100,
            block: 128
        }))
    ));
}
