// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::time::Instant;

use rusticide_model::Model;

use crate::{BenchError, Plan, Sample, throughput};

pub const DECODE_PROMPT: &str = "The history of the Roman Empire";
pub const DECODE_TOKENS: usize = 64;
/// Changing this text invalidates every recorded baseline.
pub const PREFILL_PROMPT: &str = "The Roman Empire grew from a small city on the Tiber into a \
power that ruled the whole Mediterranean world. Its legions built roads, bridges and \
aqueducts that tied distant provinces to the capital, while its lawyers wrote codes that \
shaped the legal systems of Europe for centuries. Trade flowed along those roads and across \
the sea, carrying grain from Egypt, olive oil from Spain, wine from Gaul and silk from lands \
far beyond the frontier. Emperors rose and fell, some remembered as wise administrators and \
others as tyrants, yet the machinery of government kept working through civil wars, plagues \
and invasions. In the end the western half faded away, but the eastern half endured for \
another thousand years.";

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |best, (index, &value)| {
            if value > best.1 { (index, value) } else { best }
        })
        .0
        .try_into()
        .unwrap_or(u32::MAX)
}

fn count(tokens: usize) -> f64 {
    f64::from(u32::try_from(tokens).unwrap_or(u32::MAX))
}

fn decode_run(model: &mut dyn Model, prompt: &[u32]) -> Result<f64, BenchError> {
    model.reset();
    let mut logits = model.forward_batch(prompt, 0)?;
    let start = Instant::now();
    for step in 0..DECODE_TOKENS {
        let token = std::hint::black_box(argmax(&logits));
        logits = model.forward(token, prompt.len() + step)?;
    }
    std::hint::black_box(&logits);
    throughput(count(DECODE_TOKENS), start.elapsed())
}

fn prefill_run(model: &mut dyn Model, prompt: &[u32]) -> Result<f64, BenchError> {
    model.reset();
    let start = Instant::now();
    let logits = model.forward_batch(prompt, 0)?;
    let elapsed = start.elapsed();
    std::hint::black_box(&logits);
    throughput(count(prompt.len()), elapsed)
}

/// Tokens per second of greedy decoding and of prefill on `model`.
///
/// # Errors
///
/// Any model error, or a failure to summarize the runs.
pub fn e2e_metrics(
    name: &str,
    model: &mut dyn Model,
    decode_prompt: &[u32],
    prefill_prompt: &[u32],
    plan: &Plan,
) -> Result<BTreeMap<String, Sample>, BenchError> {
    let decode = plan.sample(|| decode_run(model, decode_prompt))?;
    let prefill = plan.sample(|| prefill_run(model, prefill_prompt))?;
    Ok(BTreeMap::from([
        (format!("e2e/{name}/decode"), decode),
        (format!("e2e/{name}/prefill"), prefill),
    ]))
}

#[cfg(test)]
#[path = "e2e_tests.rs"]
mod tests;
