// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{Argument, Arguments};
use crate::ParseError;

fn round_trip(s: &str) {
    let a = Argument::parse(s).expect("parses");
    assert_eq!(a.to_string(), s);
}

#[test]
fn a_plain_argument_has_a_type_and_a_name() {
    let a = Argument::parse("Tensor self").expect("parses");
    assert_eq!(a.name, "self");
    assert!(a.annotation.is_none());
    assert!(a.default.is_none());
    round_trip("Tensor self");
}

#[test]
fn an_annotated_mutable_argument_carries_its_alias_set() {
    let a = Argument::parse("Tensor(a!) self").expect("parses");
    let ann = a.annotation.as_ref().expect("has annotation");
    assert!(ann.is_write);
    round_trip("Tensor(a!) self");
}

#[test]
fn defaults_round_trip_verbatim() {
    round_trip("bool non_blocking=False");
    round_trip("int dim=0");
    round_trip("int[1] dim=[0]");
    round_trip("ScalarType? dtype=None");
}

#[test]
fn kwarg_only_arguments_follow_the_star() {
    let args = Arguments::parse("Tensor self, *, int dim=0").expect("parses");
    assert_eq!(args.positional.len(), 1);
    assert_eq!(args.kwarg_only.len(), 1);
    assert_eq!(args.to_string(), "Tensor self, *, int dim=0");
}

#[test]
fn out_arguments_are_kwarg_only_and_annotated() {
    let args = Arguments::parse("Tensor self, *, Tensor(a!) out").expect("parses");
    assert_eq!(args.out.len(), 1);
}

/// A naive `split(',')` over the argument list severs `int[2]
/// dim=[-2,-1]` mid-value, producing two bogus arguments (`int[2]
/// dim=[-2` and `-1]`) instead of one. `-2,-1` and `0,1` are both real
/// defaults in `native_functions.yaml`
/// (`pytorch/pytorch @ 43f8f31f6a2`); `split_top_level` must be
/// bracket-depth aware to keep them intact.
#[test]
fn a_default_containing_a_comma_stays_one_argument() {
    let args = Arguments::parse("int[2] dim=[-2,-1]").expect("parses");
    assert_eq!(args.positional.len(), 1);
    assert_eq!(args.positional[0].name, "dim");
    assert_eq!(args.positional[0].default.as_deref(), Some("[-2,-1]"));
    assert_eq!(args.to_string(), "int[2] dim=[-2,-1]");
}

/// A real argument list from `native_functions.yaml`
/// (`pytorch/pytorch @ 43f8f31f6a2`), exercising kwarg-only, an
/// annotated out argument and a default together.
#[test]
fn a_corpus_argument_list_splits_and_round_trips() {
    let s = "Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out";
    let args = Arguments::parse(s).expect("parses");
    assert_eq!(args.positional.len(), 2);
    assert_eq!(args.positional[0].name, "self");
    assert_eq!(args.positional[1].name, "other");
    assert_eq!(args.kwarg_only.len(), 1);
    assert_eq!(args.kwarg_only[0].name, "alpha");
    assert_eq!(args.kwarg_only[0].default.as_deref(), Some("1"));
    assert_eq!(args.out.len(), 1);
    assert_eq!(args.out[0].name, "out");
    assert!(args.out[0].is_write());
    assert_eq!(args.to_string(), s);
}

#[test]
fn a_blank_argument_list_has_no_arguments() {
    for s in ["", "   "] {
        assert_eq!(Arguments::parse(s), Ok(Arguments::default()), "{s:?}");
    }
}

/// `)` before `(` used to build a reversed slice range and panic.
#[test]
fn a_close_paren_before_an_open_paren_is_a_bad_schema() {
    assert_eq!(
        Argument::parse("a)b( self"),
        Err(ParseError::BadSchema("a)b(".to_owned()))
    );
}

#[test]
fn a_missing_space_between_type_and_name_is_a_bad_schema() {
    assert_eq!(
        Argument::parse("Tensor"),
        Err(ParseError::BadSchema("Tensor".to_owned()))
    );
}

#[test]
fn an_unclosed_annotation_is_a_bad_schema() {
    assert_eq!(
        Argument::parse("Tensor(a! self"),
        Err(ParseError::BadSchema("Tensor(a!".to_owned()))
    );
}

#[test]
fn a_malformed_argument_fails_the_whole_argument_list() {
    assert_eq!(
        Arguments::parse("Tensor self, Tensor"),
        Err(ParseError::BadSchema("Tensor".to_owned()))
    );
}

#[test]
fn an_annotation_precedes_the_list_suffix_when_printed() {
    round_trip("Tensor(a!)[] self");
}

#[test]
fn an_annotation_precedes_the_optional_suffix_when_printed() {
    round_trip("Tensor(a)? x");
}
