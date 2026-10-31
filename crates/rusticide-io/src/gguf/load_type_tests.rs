// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use half::f16;

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
