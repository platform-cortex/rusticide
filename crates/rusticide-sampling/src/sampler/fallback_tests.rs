// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn a_draw_past_the_rounded_total_falls_back_to_the_last_survivor() {
    let mut logits = vec![0.0f32; 8014];
    logits.push(f32::NAN);
    let total = softmax(&logits[..8014])
        .iter()
        .fold(0.0f32, |sum, &p| sum + p);
    assert!(total < 1.0);
    let seed = (0..1_000_000_u64)
        .find(|&seed| fastrand::Rng::with_seed(seed).f32() >= total)
        .unwrap();
    let token = Sampler::new(seed)
        .sample_logits(
            &logits,
            &SamplingParams {
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
                repetition_penalty: 1.0,
                seed: 0,
            },
        )
        .unwrap();
    assert_eq!(token, 8013);
}
