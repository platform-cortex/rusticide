// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::batch::for_each_row;

/// Rewrites a [`rusticide_kernels::matmul_batch`] output, `by_row[row *
/// tokens + token]`, as one row per token into `by_token`.
pub(crate) fn to_token_major(by_row: &[f32], tokens: usize, by_token: &mut [f32]) {
    if tokens == 1 {
        // The decode path: both layouts agree, and a plain copy is several
        // times faster than the strided gather below.
        by_token
            .iter_mut()
            .zip(by_row)
            .for_each(|(slot, &value)| *slot = value);
        return;
    }
    let Some(rows) = by_row.len().checked_div(tokens).filter(|&rows| rows > 0) else {
        return;
    };
    for_each_row(by_token, rows, tokens, |token, slots| {
        for (slot, values) in slots.iter_mut().zip(by_row.chunks_exact(tokens)) {
            *slot = values[token];
        }
    });
}

#[cfg(test)]
mod tests {
    use super::to_token_major;

    #[test]
    fn rows_of_tokens_become_tokens_of_rows() {
        let by_row = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut by_token = [0.0; 6];
        to_token_major(&by_row, 3, &mut by_token);
        assert_eq!(by_token, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        to_token_major(&by_row, 2, &mut by_token);
        assert_eq!(by_token, [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
        to_token_major(&by_row, 1, &mut by_token);
        assert_eq!(by_token, by_row);
    }

    #[test]
    fn large_batches_match_the_serial_gather() {
        let (tokens, rows) = (64, 96);
        let by_row: Vec<f32> = (0..tokens * rows)
            .map(|index| f32::from(u16::try_from(index).unwrap()))
            .collect();
        let mut expected = vec![0.0; tokens * rows];
        for row in 0..rows {
            for token in 0..tokens {
                expected[token * rows + row] = by_row[row * tokens + token];
            }
        }
        let mut by_token = vec![0.0; tokens * rows];
        to_token_major(&by_row, tokens, &mut by_token);
        assert_eq!(by_token, expected);
    }

    #[test]
    fn zero_tokens_write_nothing() {
        let mut by_token = [7.0; 2];
        to_token_major(&[1.0, 2.0], 0, &mut by_token);
        assert_eq!(by_token, [7.0; 2]);
    }
}
