// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::Bigram;

const fn bigram(score: f32, left: usize) -> Bigram {
    Bigram {
        score,
        left,
        right: left + 1,
        size: 2,
    }
}

#[test]
fn bigram_equality_agrees_with_its_ordering() {
    assert!(bigram(-1.0, 3) == bigram(-1.0, 3));
    assert!(bigram(-1.0, 3) != bigram(-1.0, 4));
    assert!(bigram(-1.0, 3) != bigram(-2.0, 3));
    assert!(bigram(-2.0, 3) != bigram(-1.0, 3));
}
