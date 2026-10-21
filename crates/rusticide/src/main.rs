// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::process::ExitCode;

fn main() -> ExitCode {
    let outcome = rusticide::run(std::env::args().skip(1));
    print!("{}", outcome.stdout);
    eprint!("{}", outcome.stderr);
    ExitCode::from(outcome.code)
}
