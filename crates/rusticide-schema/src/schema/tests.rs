// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::FunctionSchema;
use crate::ParseError;

fn parse(s: &str) -> FunctionSchema {
    FunctionSchema::parse(s).expect("parses")
}

#[test]
fn a_single_return_round_trips() {
    let s = parse("add.Tensor(Tensor self, Tensor other) -> Tensor");
    assert_eq!(s.name.to_string(), "add.Tensor");
    assert_eq!(s.returns.len(), 1);
    assert_eq!(
        s.to_string(),
        "add.Tensor(Tensor self, Tensor other) -> Tensor"
    );
}

#[test]
fn an_empty_return_list_round_trips() {
    let s = parse("record(Tensor self) -> ()");
    assert_eq!(s.returns.len(), 0);
    assert_eq!(s.to_string(), "record(Tensor self) -> ()");
}

#[test]
fn a_named_tuple_return_round_trips() {
    let text = "max.dim(Tensor self, int dim) -> (Tensor values, Tensor indices)";
    let s = parse(text);
    assert_eq!(s.returns.len(), 2);
    assert_eq!(s.returns[0].name.as_deref(), Some("values"));
    assert_eq!(s.to_string(), text);
}

#[test]
fn an_inplace_schema_is_mutable() {
    assert!(parse("add_(Tensor(a!) self, Tensor other) -> Tensor(a!)").is_mutable());
}

#[test]
fn an_out_variant_is_mutable() {
    let text = "add.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)";
    assert!(parse(text).is_mutable());
}

#[test]
fn a_functional_schema_is_not_mutable() {
    assert!(!parse("add(Tensor self, Tensor other) -> Tensor").is_mutable());
}

#[test]
fn a_missing_open_paren_is_a_bad_schema() {
    assert_eq!(
        FunctionSchema::parse("add -> Tensor"),
        Err(ParseError::BadSchema("add -> Tensor".to_owned()))
    );
}

#[test]
fn a_missing_arrow_is_a_bad_schema() {
    assert_eq!(
        FunctionSchema::parse("add(Tensor self)"),
        Err(ParseError::BadSchema("add(Tensor self)".to_owned()))
    );
}

#[test]
fn an_arrow_before_the_open_paren_is_a_bad_schema() {
    assert_eq!(
        FunctionSchema::parse("a) -> b("),
        Err(ParseError::BadSchema("a) -> b(".to_owned()))
    );
}

#[test]
fn an_argument_error_propagates() {
    assert_eq!(
        FunctionSchema::parse("f(Bogus x) -> Tensor"),
        Err(ParseError::UnknownType("Bogus".to_owned()))
    );
}

#[test]
fn a_return_error_propagates() {
    assert_eq!(
        FunctionSchema::parse("f(Tensor x) -> Bogus"),
        Err(ParseError::UnknownType("Bogus".to_owned()))
    );
}

#[test]
fn a_non_canonical_spelling_is_a_round_trip_error() {
    assert_eq!(
        FunctionSchema::parse("f(Tensor x) -> (Tensor)"),
        Err(ParseError::RoundTrip {
            parsed: "f(Tensor x) -> Tensor".to_owned(),
            expected: "f(Tensor x) -> (Tensor)".to_owned(),
        })
    );
}
