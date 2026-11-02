// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use anyhow::Context;
use clap::Parser;
use rusticide_bench::{
    Baseline, BenchError, DECODE_PROMPT, PREFILL_PROMPT, Plan, Results, compare, e2e_metrics,
    kernel_metrics, passed, rows_to_markdown,
};
use rusticide_io::GgufFile;
use rusticide_tokenizer::Tokenizer;

const REFERENCE_MODELS: [(&str, &str); 2] = [
    ("tinyllama", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"),
    ("qwen2.5", "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
];
const MISSING_BASELINE: u8 = 2;

#[derive(Parser)]
#[command(about = "Local performance gate for rusticide")]
enum Command {
    /// Measure kernels and, for each reference model found, tokens/s.
    Run {
        #[arg(long)]
        machine: String,
        #[arg(long)]
        models: Option<PathBuf>,
        #[arg(long, default_value_t = 5)]
        runs: usize,
        #[arg(long, default_value_t = 200)]
        budget_ms: u64,
        #[arg(long)]
        out: PathBuf,
    },
    /// Print a results file as a markdown table.
    Report {
        #[arg(long)]
        results: PathBuf,
    },
    /// Compare results with a baseline; exit 1 on a regression, 2 when the
    /// baseline is missing or was recorded with another kernel adapter.
    Gate {
        #[arg(long)]
        results: PathBuf,
        #[arg(long)]
        baseline: PathBuf,
    },
    /// Record results as the baseline.
    Baseline {
        /// One results file per capture session; repeat the flag.
        #[arg(long = "results", required = true)]
        sessions: Vec<PathBuf>,
        #[arg(long)]
        out: PathBuf,
    },
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> anyhow::Result<T> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("invalid json in {}", path.display()))
}

fn write_json<T: serde::Serialize>(path: &Path, value: &T) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut text = serde_json::to_string_pretty(value)?;
    text.push('\n');
    std::fs::write(path, text).with_context(|| format!("cannot write {}", path.display()))
}

fn measure_model(
    name: &str,
    file: &Path,
    plan: &Plan,
) -> anyhow::Result<std::collections::BTreeMap<String, rusticide_bench::Sample>> {
    let source = GgufFile::load(file).with_context(|| format!("cannot load {}", file.display()))?;
    let tokenizer = Tokenizer::from_gguf(source.metadata())?;
    let mut model = rusticide_llama::registry().load(&source)?;
    let decode = tokenizer.encode(DECODE_PROMPT)?;
    let prefill = tokenizer.encode(PREFILL_PROMPT)?;
    Ok(e2e_metrics(name, model.as_mut(), &decode, &prefill, plan)?)
}

fn run(
    machine: String,
    models: Option<&Path>,
    plan: &Plan,
    out: &Path,
) -> anyhow::Result<ExitCode> {
    let mut metrics = kernel_metrics(plan)?;
    for (name, file) in REFERENCE_MODELS {
        let path = models
            .map(|dir| dir.join(file))
            .filter(|path| path.exists());
        if let Some(path) = path {
            metrics.extend(measure_model(name, &path, plan)?);
        }
    }
    let adapter = rusticide_kernels::active().name().to_owned();
    write_json(
        out,
        &Results {
            machine,
            adapter,
            metrics,
        },
    )?;
    Ok(ExitCode::SUCCESS)
}

fn gate(results: &Path, baseline: &Path) -> anyhow::Result<ExitCode> {
    if !baseline.exists() {
        eprintln!(
            "no baseline at {}; record one with `mise run bench:baseline`",
            baseline.display()
        );
        return Ok(ExitCode::from(MISSING_BASELINE));
    }
    let rows = match compare(&read_json(baseline)?, &read_json(results)?) {
        Err(error @ BenchError::OtherAdapter { .. }) => {
            eprintln!("{error}; record one with `mise run bench:baseline`");
            return Ok(ExitCode::from(MISSING_BASELINE));
        }
        rows => rows?,
    };
    print!("{}", rows_to_markdown(&rows));
    Ok(if passed(&rows) {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

fn dispatch(command: Command) -> anyhow::Result<ExitCode> {
    match command {
        Command::Run {
            machine,
            models,
            runs,
            budget_ms,
            out,
        } => {
            let plan = Plan {
                runs,
                budget: Duration::from_millis(budget_ms),
            };
            run(machine, models.as_deref(), &plan, &out)
        }
        Command::Report { results } => {
            print!("{}", read_json::<Results>(&results)?.to_markdown());
            Ok(ExitCode::SUCCESS)
        }
        Command::Gate { results, baseline } => gate(&results, &baseline),
        Command::Baseline { sessions, out } => {
            let sessions = sessions
                .iter()
                .map(|path| read_json(path))
                .collect::<anyhow::Result<Vec<Results>>>()?;
            write_json(&out, &Baseline::from_sessions(&sessions)?)?;
            Ok(ExitCode::SUCCESS)
        }
    }
}

fn main() -> ExitCode {
    dispatch(Command::parse()).unwrap_or_else(|error| {
        eprintln!("error: {error:#}");
        ExitCode::FAILURE
    })
}
