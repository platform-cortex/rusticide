// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The clap command line behind the `cli` feature.

mod cmd;

use std::ffi::OsString;
use std::io::Write;

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
    match execute(cli.command, out) {
        Ok(()) => 0,
        Err(error) => {
            let _ = writeln!(err, "error: {error:#}");
            1
        }
    }
}

fn execute(command: Command, out: &mut impl Write) -> anyhow::Result<()> {
    match command {
        Command::Schema { schema } => cmd::schema::run(&schema, out),
        Command::Version => Ok(cmd::version::run(out)?),
    }
}

#[cfg(test)]
mod tests;
