// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::process::ExitCode;

fn main() -> ExitCode {
    ExitCode::from(rusticide::cli::run(
        std::env::args(),
        &mut std::io::stdout().lock(),
        &mut std::io::stderr().lock(),
    ))
}
