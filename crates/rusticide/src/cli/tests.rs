// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::run;

const ADD: &str = "add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor";

fn invoke(args: &[&str]) -> (u8, String, String) {
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let code = run(args, &mut out, &mut err);
    (
        code,
        String::from_utf8(out).unwrap(),
        String::from_utf8(err).unwrap(),
    )
}

fn version_line() -> String {
    format!("rusticide {}\n", env!("CARGO_PKG_VERSION"))
}

#[test]
fn version_flags_print_exactly_the_version_line() {
    for flag in ["--version", "-V"] {
        assert_eq!(
            invoke(&["rusticide", flag]),
            (0, version_line(), String::new()),
            "{flag}"
        );
    }
}

#[test]
fn the_version_subcommand_prints_the_same_line() {
    assert_eq!(
        invoke(&["rusticide", "version"]),
        (0, version_line(), String::new())
    );
}

#[test]
fn help_lists_the_subcommands_on_stdout() {
    let (code, out, err) = invoke(&["rusticide", "--help"]);
    assert_eq!(code, 0);
    assert!(
        out.contains("schema") && out.contains("version") && out.contains("run"),
        "{out}"
    );
    assert_eq!(err, "");
}

/// Clap's `arg_required_else_help`: usage goes to stderr with the usage-error code.
#[test]
fn no_arguments_prints_help_to_stderr_with_code_2() {
    let (code, out, err) = invoke(&["rusticide"]);
    assert_eq!(code, 2);
    assert_eq!(out, "");
    assert!(err.contains("Usage: rusticide"), "{err}");
}

#[test]
fn an_unknown_flag_is_a_usage_error_on_stderr() {
    let (code, out, err) = invoke(&["rusticide", "--frobnicate"]);
    assert_eq!(code, 2);
    assert_eq!(out, "");
    assert!(err.contains("--frobnicate"), "{err}");
}

#[test]
fn the_schema_subcommand_prints_the_parsed_structure() {
    let (code, out, err) = invoke(&["rusticide", "schema", ADD]);
    assert_eq!(code, 0);
    assert!(out.ends_with(&format!("round-trip: {ADD}\n")), "{out}");
    assert_eq!(err, "");
}

#[test]
fn an_invalid_schema_fails_with_the_parse_error_on_stderr() {
    let (code, out, err) = invoke(&["rusticide", "schema", "not a schema"]);
    assert_eq!(code, 1);
    assert_eq!(out, "");
    assert_eq!(err, "error: malformed schema: not a schema\n");
}

#[test]
fn run_reports_a_missing_file() {
    let (code, out, err) = invoke(&["rusticide", "run", "/nonexistent.gguf", "--prompt", "x"]);
    assert_eq!(code, 1);
    assert_eq!(out, "");
    assert!(err.starts_with("error: "), "{err}");
}

#[test]
fn run_reports_a_non_gguf_file() {
    let path = std::env::temp_dir().join(format!("rusticide-junk-{}.gguf", std::process::id()));
    std::fs::write(&path, b"this is not a gguf file").unwrap();
    let path_text = path.to_str().unwrap().to_owned();
    let (code, out, err) = invoke(&["rusticide", "run", &path_text, "--prompt", "x"]);
    std::fs::remove_file(&path).unwrap();
    assert_eq!(code, 1);
    assert_eq!(out, "");
    assert!(err.starts_with("error: cannot load "), "{err}");
}

#[test]
fn run_requires_a_prompt() {
    let (code, out, err) = invoke(&["rusticide", "run", "model.gguf"]);
    assert_eq!(code, 2);
    assert_eq!(out, "");
    assert!(err.contains("--prompt"), "{err}");
}
