// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Round-trip every operator schema `PyTorch` declares.
//!
//! This is the acceptance test for the parser: parse, re-print, require
//! byte equality. A schema that does not round-trip is a schema whose
//! representation lost information.

use rusticide_schema::FunctionSchema;

const CORPUS: &str = include_str!("data/schemas.txt");

#[test]
fn every_schema_round_trips_byte_exactly() {
    let mut failures = Vec::new();
    let mut count = 0usize;
    for line in CORPUS.lines().map(str::trim).filter(|l| !l.is_empty()) {
        count += 1;
        match FunctionSchema::parse(line) {
            Ok(s) => {
                let printed = s.to_string();
                if printed != line {
                    failures.push(format!("  got: {printed}\n  want: {line}"));
                }
            }
            Err(e) => failures.push(format!("  {line}\n  error: {e}")),
        }
    }
    assert!(count >= 2585, "corpus shrank: only {count} schemas");
    assert!(
        failures.is_empty(),
        "{} of {count} schemas failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn the_corpus_contains_the_annotation_shapes_we_rely_on() {
    let mutable = CORPUS.lines().filter(|l| l.contains("(a!)")).count();
    assert!(
        mutable > 900,
        "expected ~981 mutable schemas, saw {mutable}"
    );
}
