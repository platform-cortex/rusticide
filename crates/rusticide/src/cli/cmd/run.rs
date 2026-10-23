// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Context;
use rusticide_io::GgufFile;
use rusticide_model::{GenerateOptions, ModelError, ModelSource, generate};
use rusticide_sampling::{Sampler, SamplingParams};
use rusticide_tokenizer::Tokenizer;

use crate::cli::registry::registry;

/// The parsed `run` arguments.
#[derive(Debug, Clone)]
pub(crate) struct RunArgs {
    pub(crate) model: PathBuf,
    pub(crate) prompt: String,
    pub(crate) max_tokens: usize,
    pub(crate) temperature: f32,
    pub(crate) top_k: u32,
    pub(crate) top_p: f32,
    pub(crate) seed: Option<u64>,
}

/// Load the model file and stream a completion of the prompt to `out`.
pub(crate) fn run(
    args: &RunArgs,
    out: &mut impl Write,
    err: &mut impl Write,
) -> anyhow::Result<()> {
    let loaded = GgufFile::load(&args.model)
        .with_context(|| format!("cannot load {}", args.model.display()));
    loaded.and_then(|source| run_source(&source, args, out, err))
}

/// Generate from an already loaded source; the stats line goes to `err`.
pub(crate) fn run_source(
    source: &dyn ModelSource,
    args: &RunArgs,
    out: &mut impl Write,
    err: &mut impl Write,
) -> anyhow::Result<()> {
    let tokenizer = Tokenizer::from_gguf(source.metadata())?;
    let mut model = registry().load(source)?;
    let seed = args.seed.unwrap_or_else(clock_seed);
    let options = GenerateOptions {
        max_tokens: args.max_tokens,
        params: SamplingParams {
            temperature: args.temperature,
            top_k: args.top_k,
            top_p: args.top_p,
            repetition_penalty: 1.0,
            seed,
        },
    };
    let mut sampler = Sampler::new(seed);
    let mut write_failure = None;
    let mut emit = |text: &str| -> Result<(), ModelError> {
        let streamed = out.write_all(text.as_bytes()).and_then(|()| out.flush());
        streamed.map_err(|error| {
            write_failure = Some(error);
            ModelError::InvalidConfig {
                reason: "output write failed",
            }
        })
    };
    let generated = generate(
        model.as_mut(),
        &tokenizer,
        &args.prompt,
        &options,
        &mut sampler,
        &mut emit,
    );
    if let Some(error) = write_failure {
        return Err(error.into());
    }
    let generation = generated?;
    out.flush()?;
    writeln!(
        err,
        "\nprompt: {} tokens, generated: {} tokens, {:.2} tokens/s",
        generation.prompt_tokens.len(),
        generation.generated_tokens.len(),
        tokens_per_second(generation.generated_tokens.len(), generation.elapsed),
    )?;
    Ok(())
}

fn clock_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| {
            u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX)
        })
}

fn tokens_per_second(tokens: usize, elapsed: Duration) -> f64 {
    let seconds = elapsed.as_secs_f64();
    if seconds == 0.0 {
        return 0.0;
    }
    f64::from(u32::try_from(tokens).unwrap_or(u32::MAX)) / seconds
}

#[cfg(test)]
#[path = "run_tests.rs"]
mod tests;
