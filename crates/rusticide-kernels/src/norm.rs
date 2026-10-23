// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::lossy;

/// Divides the row by `sqrt(mean(x^2) + eps)`. An empty row is a no-op.
pub fn rms_norm(row: &mut [f32], eps: f32) {
    let mean_square = row.iter().map(|&value| value * value).sum::<f32>() / lossy(row.len());
    let rms = (mean_square + eps).sqrt();
    for value in row.iter_mut() {
        *value /= rms;
    }
}

/// Max-subtracted softmax in place. An empty row is a no-op.
pub fn softmax(row: &mut [f32]) {
    let max = row
        .iter()
        .fold(f32::NEG_INFINITY, |acc, &value| acc.max(value));
    let mut sum = 0.0;
    for value in row.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    for value in row.iter_mut() {
        *value /= sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_divides_by_root_mean_square() {
        let mut row = [1.0, 2.0, 3.0, 4.0];
        rms_norm(&mut row, 0.0);
        let rms = 7.5_f32.sqrt();
        assert_eq!(row, [1.0 / rms, 2.0 / rms, 3.0 / rms, 4.0 / rms]);
    }

    #[test]
    fn rms_norm_of_empty_row_is_a_no_op() {
        let mut row: [f32; 0] = [];
        rms_norm(&mut row, 1e-5);
        assert_eq!(row, []);
    }

    #[test]
    fn softmax_sums_to_one_and_survives_large_inputs() {
        let mut row = [1000.0, 1000.0];
        softmax(&mut row);
        assert_eq!(row, [0.5, 0.5]);
    }

    #[test]
    fn softmax_of_empty_row_is_a_no_op() {
        let mut row: [f32; 0] = [];
        softmax(&mut row);
        assert_eq!(row, []);
    }
}
