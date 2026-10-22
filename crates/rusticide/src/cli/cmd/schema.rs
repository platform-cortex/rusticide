// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::io::Write;

use anyhow::Result;
use rusticide_schema::FunctionSchema;

/// Parse an operator schema and print its structure.
pub(crate) fn run(schema: &str, out: &mut impl Write) -> Result<()> {
    let parsed = FunctionSchema::parse(schema)?;
    writeln!(out, "name:       {}", parsed.name)?;
    writeln!(out, "mutable:    {}", parsed.is_mutable())?;
    writeln!(out, "arguments:  {}", parsed.arguments)?;
    writeln!(out, "round-trip: {parsed}")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::run;

    const ADD: &str = "add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor";

    fn render(schema: &str) -> anyhow::Result<String> {
        let mut out = Vec::new();
        run(schema, &mut out)?;
        Ok(String::from_utf8(out)?)
    }

    #[test]
    fn prints_the_four_lines_and_echoes_the_input() {
        let expected = [
            "name:       add.Tensor",
            "mutable:    false",
            "arguments:  Tensor self, Tensor other, *, Scalar alpha=1",
            &format!("round-trip: {ADD}"),
            "",
        ]
        .join("\n");
        assert_eq!(render(ADD).unwrap(), expected);
    }

    #[test]
    fn a_mutating_schema_reports_mutable() {
        let out = render("add_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)").unwrap();
        assert!(out.contains("mutable:    true\n"), "{out}");
    }

    #[test]
    fn an_invalid_schema_is_an_error() {
        let err = render("not a schema").unwrap_err();
        assert_eq!(err.to_string(), "malformed schema: not a schema");
    }
}
