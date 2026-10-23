// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The clap command line behind the `cli` feature.

mod cmd;
mod registry;

use std::ffi::OsString;
use std::io::Write;
use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// The `rusticide` command line.
#[derive(Debug, Parser)]
#[command(name = "rusticide", version, arg_required_else_help = true)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Parse an operator schema and print its structure
    Schema {
        /// The schema, e.g. `add.Tensor(Tensor self, Tensor other) -> Tensor`
        schema: String,
    },
    /// Generate text from a GGUF model
    Run {
        /// The GGUF model file
        model: PathBuf,
        /// The prompt to complete
        #[arg(long)]
        prompt: String,
        /// The maximum number of tokens to generate
        #[arg(short = 'n', long, default_value_t = 128)]
        max_tokens: usize,
        /// The sampling temperature; 0 means greedy
        #[arg(long, default_value_t = 0.8)]
        temperature: f32,
        /// Keep only the k most likely tokens
        #[arg(long, default_value_t = 40)]
        top_k: u32,
        /// Nucleus sampling probability mass
        #[arg(long, default_value_t = 0.95)]
        top_p: f32,
        /// The sampler seed; random when absent
        #[arg(long)]
        seed: Option<u64>,
    },
    /// Print the version
    Version,
}

/// Run the CLI against `args` (program name first), writing to the given streams.
///
/// Help and version requests go to `out` with code 0; usage errors, including
/// no arguments at all (clap prints the help as an error), go to `err` with
/// code 2; a failing command prints `error: <cause>` to `err` with code 1.
pub fn run<I, T>(args: I, out: &mut impl Write, err: &mut impl Write) -> u8
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let cli = match Cli::try_parse_from(args) {
        Ok(cli) => cli,
        Err(error) if error.use_stderr() => {
            let _ = write!(err, "{}", error.render());
            return 2;
        }
        Err(error) => {
            let _ = write!(out, "{}", error.render());
            return 0;
        }
    };
    match execute(cli.command, out, err) {
        Ok(()) => 0,
        Err(error) => {
            let _ = writeln!(err, "error: {error:#}");
            1
        }
    }
}

fn execute(command: Command, out: &mut impl Write, err: &mut impl Write) -> anyhow::Result<()> {
    match command {
        Command::Run {
            model,
            prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
            seed,
        } => cmd::run::run(
            &cmd::run::RunArgs {
                model,
                prompt,
                max_tokens,
                temperature,
                top_k,
                top_p,
                seed,
            },
            out,
            err,
        ),
        Command::Schema { schema } => cmd::schema::run(&schema, out),
        Command::Version => Ok(cmd::version::run(out)?),
    }
}

#[cfg(test)]
mod tests;
