// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_kernels::{mul_assign, rms_norm};

use crate::batch::zip_rows;

/// Writes `rms_norm(input) * weight` into `output`.
pub fn normalize(input: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    output.copy_from_slice(input);
    rms_norm(output, eps);
    mul_assign(output, weight);
}

/// [`normalize`] applied to each `weight`-wide row of `input`.
pub fn normalize_rows(input: &[f32], weight: &[f32], eps: f32, tokens: usize, output: &mut [f32]) {
    zip_rows(output, input, weight.len(), tokens, |out, row| {
        normalize(row, weight, eps, out);
    });
}

#[cfg(test)]
mod tests {
    use super::{normalize, normalize_rows};

    #[test]
    fn normalize_scales_to_unit_rms_then_applies_the_weight() {
        let mut output = [0.0; 2];
        normalize(&[3.0, 4.0], &[1.0, 2.0], 0.0, &mut output);
        let rms = 12.5_f32.sqrt();
        assert!((output[0] - 3.0 / rms).abs() < 1e-6);
        assert!((output[1] - 2.0 * 4.0 / rms).abs() < 1e-6);
    }

    #[test]
    fn normalize_rows_normalizes_each_row_alone() {
        for tokens in [1, 3] {
            let input: Vec<f32> = [3.0, 4.0].repeat(tokens);
            let mut rows = vec![0.0; 2 * tokens];
            normalize_rows(&input, &[1.0, 2.0], 0.0, tokens, &mut rows);
            let mut single = [0.0; 2];
            normalize(&[3.0, 4.0], &[1.0, 2.0], 0.0, &mut single);
            for row in rows.as_chunks::<2>().0 {
                assert_eq!(*row, single);
            }
        }
    }
}
