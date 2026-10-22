// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use half::f16;
use rusticide_core::DType;
use rusticide_quantize::QuantizeError;

use super::GgufFile;
use crate::error::GgufError;
use crate::gguf::test_support::GgufBuilder;

const F32: u32 = 0;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn file(builder: GgufBuilder) -> GgufFile {
    GgufFile::from_bytes(builder.build()).unwrap()
}

#[test]
fn tensor_not_found() {
    let file = file(GgufBuilder::new());
    assert!(matches!(
        file.tensor("nonexistent"),
        Err(GgufError::TensorNotFound(name)) if name == "nonexistent"
    ));
}

#[test]
fn tensor_by_index_out_of_range() {
    let file = file(GgufBuilder::new().tensor("a", &[1], F32, &f32_bytes(&[1.0])));
    assert!(matches!(
        file.tensor_by_index(1),
        Err(GgufError::TensorNotFound(name)) if name == "index 1"
    ));
}

#[test]
fn f32_tensor_data() {
    let values: Vec<f32> = (0..12_u8).map(f32::from).collect();
    let file = file(GgufBuilder::new().tensor("w", &[4, 3], F32, &f32_bytes(&values)));
    let tensor = file.tensor("w").unwrap();
    assert_eq!(tensor.dtype(), DType::F32);
    assert_eq!(tensor.numel(), 12);
    assert_eq!(tensor.shape().dims(), [3, 4]);
    assert_eq!(
        tensor.get::<f32>(&[0, 0]).unwrap().to_bits(),
        0.0_f32.to_bits()
    );
    assert_eq!(
        tensor.get::<f32>(&[2, 3]).unwrap().to_bits(),
        11.0_f32.to_bits()
    );
    assert_eq!(tensor.to_vec::<f32>().unwrap(), values);
}

#[test]
fn tensor_1d() {
    let file = file(GgufBuilder::new().tensor("v", &[3], F32, &f32_bytes(&[1.5, -2.5, 3.5])));
    assert_eq!(
        file.tensor("v").unwrap().to_vec::<f32>().unwrap(),
        [1.5, -2.5, 3.5]
    );
}

#[test]
fn tensor_by_index() {
    let file = file(
        GgufBuilder::new()
            .tensor("b", &[1], F32, &f32_bytes(&[2.0]))
            .tensor("a", &[1], F32, &f32_bytes(&[1.0])),
    );
    assert_eq!(
        file.tensor_by_index(0).unwrap().to_vec::<f32>().unwrap(),
        [1.0]
    );
    assert_eq!(
        file.tensor_by_index(1).unwrap().to_vec::<f32>().unwrap(),
        [2.0]
    );
}

#[test]
fn a_byte_length_that_overflows_is_a_cast_overflow() {
    let bytes = GgufBuilder::new().tensor("w", &[1 << 62], F32, &[]).build();
    assert!(matches!(
        GgufFile::from_bytes(bytes).unwrap().tensor("w"),
        Err(GgufError::CastOverflow("tensor byte length"))
    ));
}

#[test]
fn f16_tensor_data() {
    let values = [f16::from_f32(1.5), f16::from_f32(-2.0), f16::from_f32(0.25)];
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let file = file(GgufBuilder::new().tensor("h", &[3], 1, &bytes));
    let tensor = file.tensor("h").unwrap();
    assert_eq!(tensor.dtype(), DType::F16);
    assert_eq!(tensor.get::<f16>(&[1]).unwrap(), f16::from_f32(-2.0));
    assert_eq!(tensor.to_vec::<f16>().unwrap(), values);
}

const F16_ONE: [u8; 2] = [0x00, 0x3C];

#[test]
fn q4_0_tensor_data() {
    let mut block = F16_ONE.to_vec();
    block.extend([0x9A; 16]);
    let file = file(GgufBuilder::new().tensor("q", &[32], 2, &block));
    let tensor = file.tensor("q").unwrap();
    assert_eq!(tensor.dtype(), DType::F32);
    assert_eq!(
        tensor.get::<f32>(&[0]).unwrap().to_bits(),
        2.0_f32.to_bits()
    );
    assert_eq!(
        tensor.get::<f32>(&[16]).unwrap().to_bits(),
        1.0_f32.to_bits()
    );
}

#[test]
fn q4_k_tensor() {
    let mut block = F16_ONE.to_vec();
    block.extend([0, 0]);
    block.extend([2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0]);
    block.extend([0x31; 128]);
    let file = file(GgufBuilder::new().tensor("k", &[256], 12, &block));
    let tensor = file.tensor("k").unwrap();
    assert_eq!(
        tensor.get::<f32>(&[0]).unwrap().to_bits(),
        2.0_f32.to_bits()
    );
    assert_eq!(
        tensor.get::<f32>(&[32]).unwrap().to_bits(),
        6.0_f32.to_bits()
    );
    assert_eq!(
        tensor.get::<f32>(&[128]).unwrap().to_bits(),
        0.0_f32.to_bits()
    );
}

#[test]
fn a_k_quant_with_an_unaligned_element_count_is_a_quantize_error() {
    let file = file(GgufBuilder::new().tensor("k", &[100], 12, &[0; 144]));
    assert!(matches!(
        file.tensor("k"),
        Err(GgufError::Quantize(QuantizeError::UnalignedShape {
            numel: 100,
            block: 256,
            ..
        }))
    ));
}

#[test]
fn q6_k_tensor() {
    let mut block = vec![0; 192];
    block.extend([1; 16]);
    block.extend(F16_ONE);
    let file = file(GgufBuilder::new().tensor("s", &[256], 14, &block));
    let tensor = file.tensor("s").unwrap();
    for index in [0, 100, 255] {
        let value = tensor.get::<f32>(&[index]).unwrap();
        assert_eq!(value.to_bits(), (-32.0_f32).to_bits(), "element {index}");
    }
}

fn patch_last_offset(bytes: &mut [u8], data_len: usize, offset: u64) {
    let at = bytes.len() - data_len - 8;
    bytes[at..at + 8].copy_from_slice(&offset.to_le_bytes());
}

fn single_unaligned_f32() -> Vec<u8> {
    GgufBuilder::new()
        .alignment(1)
        .tensor("w", &[1], F32, &f32_bytes(&[1.0]))
        .build()
}

#[test]
fn q4_0_short_data() {
    let block = [0_u8; 72];
    let mut bytes = GgufBuilder::new().tensor("q", &[100], 2, &block).build();
    let file_len = bytes.len();
    bytes.truncate(file_len - 40);
    let file = GgufFile::from_bytes(bytes).unwrap();
    assert!(matches!(file.tensor("q"), Err(GgufError::UnexpectedEof)));
}

#[test]
fn offset_that_overflows_is_eof() {
    let mut bytes = single_unaligned_f32();
    patch_last_offset(&mut bytes, 4, u64::MAX - 1);
    let file = GgufFile::from_bytes(bytes).unwrap();
    assert!(matches!(file.tensor("w"), Err(GgufError::UnexpectedEof)));
}

#[test]
fn an_end_that_overflows_is_eof() {
    let mut bytes = single_unaligned_f32();
    let data_start = bytes.len() as u64 - 4;
    patch_last_offset(&mut bytes, 4, u64::MAX - data_start - 1);
    let file = GgufFile::from_bytes(bytes).unwrap();
    assert!(matches!(file.tensor("w"), Err(GgufError::UnexpectedEof)));
}

#[test]
fn offset_is_relative_to_the_data_section() {
    let file = file(
        GgufBuilder::new()
            .tensor("first", &[2], F32, &f32_bytes(&[1.0, 2.0]))
            .tensor("second", &[2], F32, &f32_bytes(&[3.0, 4.0])),
    );
    assert_eq!(file.tensor_info("second").unwrap().offset, 32);
    assert_eq!(
        file.tensor("second").unwrap().to_vec::<f32>().unwrap(),
        [3.0, 4.0]
    );
}

#[test]
fn multiple_tensors_data() {
    let file = file(
        GgufBuilder::new()
            .tensor("a", &[2], F32, &f32_bytes(&[1.0, 2.0]))
            .tensor("b", &[1], 1, &f16::from_f32(8.0).to_le_bytes())
            .tensor("c", &[3], F32, &f32_bytes(&[5.0, 6.0, 7.0])),
    );
    assert_eq!(
        file.tensor("a").unwrap().to_vec::<f32>().unwrap(),
        [1.0, 2.0]
    );
    assert_eq!(
        file.tensor("b").unwrap().to_vec::<f16>().unwrap(),
        [f16::from_f32(8.0)]
    );
    assert_eq!(
        file.tensor("c").unwrap().to_vec::<f32>().unwrap(),
        [5.0, 6.0, 7.0]
    );
}

#[test]
fn round_trip_binary() {
    let path = std::env::temp_dir().join(format!("rusticide-io-load-{}.gguf", std::process::id()));
    let builder = GgufBuilder::new().tensor("w", &[2], F32, &f32_bytes(&[0.5, -0.5]));
    std::fs::write(&path, builder.build()).unwrap();
    let loaded = GgufFile::load(&path);
    std::fs::remove_file(&path).unwrap();
    assert_eq!(
        loaded
            .unwrap()
            .tensor("w")
            .unwrap()
            .to_vec::<f32>()
            .unwrap(),
        [0.5, -0.5]
    );
}
