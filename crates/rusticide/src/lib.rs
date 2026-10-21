// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `rusticide` command-line entry point.
//!
//! All behavior lives here so it is unit-tested; `main.rs` only forwards
//! `std::env::args` and writes the [`Outcome`] out.

/// What one invocation produced: an exit code plus the text for each stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Outcome {
    pub code: u8,
    pub stdout: String,
    pub stderr: String,
}

/// Run the CLI against `args`, excluding the program name.
#[must_use]
pub fn run<I, S>(args: I) -> Outcome
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let first = args.into_iter().next();
    match first.as_ref().map(AsRef::as_ref) {
        None | Some("--help" | "-h") => success(USAGE.to_owned()),
        Some("--version" | "-V") => success(format!("rusticide {}\n", env!("CARGO_PKG_VERSION"))),
        Some(other) => Outcome {
            code: 2,
            stdout: String::new(),
            stderr: format!("error: unrecognized argument '{other}'\n\n{USAGE}"),
        },
    }
}

const USAGE: &str = "\
Usage: rusticide [OPTIONS]

Options:
  -h, --help     Print this help
  -V, --version  Print the version
";

const fn success(stdout: String) -> Outcome {
    Outcome {
        code: 0,
        stdout,
        stderr: String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::run;

    #[test]
    fn version_flag_prints_the_crate_version() {
        for flag in ["--version", "-V"] {
            let out = run([flag]);
            assert_eq!(out.code, 0, "{flag}");
            assert_eq!(
                out.stdout,
                format!("rusticide {}\n", env!("CARGO_PKG_VERSION"))
            );
            assert_eq!(out.stderr, "");
        }
    }

    #[test]
    fn help_flag_prints_usage_to_stdout() {
        for flag in ["--help", "-h"] {
            let out = run([flag]);
            assert_eq!(out.code, 0, "{flag}");
            assert!(out.stdout.starts_with("Usage: rusticide"), "{}", out.stdout);
            assert_eq!(out.stderr, "");
        }
    }

    #[test]
    fn no_arguments_prints_usage() {
        let out = run(std::iter::empty::<&str>());
        assert_eq!(out.code, 0);
        assert!(out.stdout.starts_with("Usage: rusticide"));
    }

    /// Exit code 2 is the conventional usage-error status (sysexits `EX_USAGE`
    /// is 64, but 2 is what clap, getopt and the shells use).
    #[test]
    fn an_unknown_argument_is_a_usage_error_on_stderr() {
        let out = run(["--frobnicate"]);
        assert_eq!(out.code, 2);
        assert_eq!(out.stdout, "");
        assert!(
            out.stderr.contains("unrecognized argument '--frobnicate'"),
            "{}",
            out.stderr
        );
        assert!(out.stderr.contains("Usage: rusticide"));
    }
}
