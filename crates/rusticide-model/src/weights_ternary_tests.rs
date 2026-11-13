// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_io::Metadata;
use rusticide_kernels::Weights;
use rusticide_quantize::block_format;

use super::{Matrix, ternary_scale};
use crate::error::ModelError;
use crate::source::InMemorySource;

const SCALE: f32 = 0.75;

/// Row 0 is all +1; row 1 holds -1, 0, +1 and +2 in its four lanes.
fn i2_s_source(scale: f32) -> InMemorySource {
    let mut data = vec![0b1010_1010; 32];
    data.extend([0b0001_1011; 32]);
    data.extend(scale.to_le_bytes());
    data.extend([0xA5; 28]);
    InMemorySource::new(Metadata::default())
        .with_quantized("b", block_format(36).unwrap(), data, &[2, 128])
        .unwrap()
}

fn lanes(values: [f32; 4]) -> Vec<f32> {
    values.iter().flat_map(|&v| [v; 32]).collect()
}

#[test]
fn an_i2_s_matrix_stays_ternary_without_its_trailer() {
    let matrix = Matrix::load(&i2_s_source(SCALE), "b", 2, 128).unwrap();
    assert!(matrix.is_quantized());
    assert_eq!(matrix.rows(), 2);
    let Weights::Ternary {
        data,
        row_bytes,
        scale,
    } = matrix.weights()
    else {
        panic!("not ternary");
    };
    assert_eq!((data.len(), row_bytes, scale), (64, 32, SCALE));
    assert_eq!(matrix.weights().rows(), 2);
}

#[test]
fn copy_row_decodes_code_minus_one_times_the_scale() {
    let matrix = Matrix::load(&i2_s_source(SCALE), "b", 2, 128).unwrap();
    let mut row = vec![0.0; 128];
    assert!(matrix.copy_row(0, &mut row));
    assert_eq!(row, lanes([SCALE; 4]));
    assert!(matrix.copy_row(1, &mut row));
    assert_eq!(row, lanes([-SCALE, 0.0, SCALE, 2.0 * SCALE]));
    assert!(!matrix.copy_row(2, &mut row));
}

/// Activations `k / 2` with `max |k| = 127`: the row divisor is exactly 2
/// and every quant is `k`; the integer dot is exact in `f32` here.
#[test]
fn a_matvec_is_the_integer_dot_over_s_times_the_scale() {
    let matrix = Matrix::load(&i2_s_source(SCALE), "b", 2, 128).unwrap();
    let quants: Vec<i16> = (0..128).map(|i| 127 - 2 * i).collect();
    let input: Vec<f32> = quants.iter().map(|&k| f32::from(k) / 2.0).collect();
    let weights = [lanes([1.0; 4]), lanes([-1.0, 0.0, 1.0, 2.0])];
    let expected: Vec<u32> = weights
        .iter()
        .map(|row| {
            let dot: f32 = row
                .iter()
                .zip(&quants)
                .map(|(&w, &q)| w * f32::from(q))
                .sum();
            (dot / 2.0 * SCALE).to_bits()
        })
        .collect();
    assert_ne!(expected[1], 0);
    let mut output = [f32::NAN; 2];
    matrix.matvec(&input, &mut output);
    assert_eq!(output.map(f32::to_bits), expected[..]);
}

#[test]
fn a_non_finite_scale_is_an_invalid_tensor() {
    for scale in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let error = Matrix::load(&i2_s_source(scale), "b", 2, 128).unwrap_err();
        assert!(matches!(
            &error,
            ModelError::InvalidTensor { name, reason: "its scale is missing or not finite" }
                if name == "b"
        ));
        assert_eq!(
            error.to_string(),
            "tensor b is invalid: its scale is missing or not finite"
        );
    }
}

#[test]
fn a_trailer_too_short_for_a_scale_is_an_invalid_tensor() {
    assert!(matches!(
        ternary_scale("t", &[0; 3]),
        Err(ModelError::InvalidTensor { name, .. }) if name == "t"
    ));
    assert_eq!(ternary_scale("t", &2.5_f32.to_le_bytes()).unwrap(), 2.5);
}
