// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Annotation`: the alias-and-mutability grammar on schema arguments.

use core::fmt;

use crate::ParseError;

/// An alias annotation, e.g. the `a!` inside `Tensor(a!)`.
///
/// `PyTorch` schemas attach these to declare, per argument, which alias set
/// it belongs to and whether the operator writes through it -- the data
/// that lets the whole operator surface be checked for aliasing bugs
/// without inspecting kernel bodies. `parse` enforces the two invariants
/// torchgen enforces at load time:
///
/// - A write may name only one alias set: `is_write && alias_set.len() >
///   1` is rejected. That is aliasing XOR mutation, the same exclusivity
///   Rust's borrow checker enforces statically (`&mut T` is never
///   shared) -- checked here dynamically, against a text corpus, instead
///   of by the type system.
/// - `alias_set` and `alias_set_after` cannot both name more than one
///   letter.
///
/// The `-> *` form (used by the `split`/`chunk`/`unbind` family) is a
/// wildcard escape: the result may alias anything, not just the named
/// set.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Annotation {
    /// The alias set this argument belongs to. Not actually a set --
    /// canonically ordered as written, per torchgen.
    pub alias_set: Vec<char>,
    /// Whether the operator writes through this argument.
    pub is_write: bool,
    /// The alias set after the operator runs, from a `-> ...` clause.
    /// `['*']` for the wildcard escape; empty when the schema has no
    /// `->` clause.
    pub alias_set_after: Vec<char>,
}

impl Annotation {
    /// Parse an annotation body -- the contents of `(...)`, without the
    /// parentheses -- asserting it round-trips.
    ///
    /// # Errors
    ///
    /// Returns [`ParseError::BadAnnotation`] if `s` does not match the
    /// annotation grammar, or violates either invariant described on
    /// [`Annotation`]. Returns [`ParseError::RoundTrip`] if the parsed
    /// value fails to print back to exactly `s` (a bug in this parser,
    /// not in the input).
    pub fn parse(s: &str) -> Result<Self, ParseError> {
        let parsed = Self::parse_inner(s)?;
        let printed = parsed.to_string();
        if printed == s {
            Ok(parsed)
        } else {
            Err(ParseError::RoundTrip {
                parsed: printed,
                expected: s.to_owned(),
            })
        }
    }

    fn parse_inner(s: &str) -> Result<Self, ParseError> {
        let bad = || ParseError::BadAnnotation(s.to_owned());
        let (before, after) = s
            .split_once(" -> ")
            .map_or((s, None), |(b, a)| (b, Some(a)));
        let (set_str, is_write) = before
            .strip_suffix('!')
            .map_or((before, false), |rest| (rest, true));

        let alias_set = parse_alias_set(set_str).ok_or_else(bad)?;
        if is_write && alias_set.len() > 1 {
            return Err(bad());
        }

        let alias_set_after = match after {
            None => Vec::new(),
            Some("*") => vec!['*'],
            Some(a) => parse_alias_set(a).ok_or_else(bad)?,
        };
        if alias_set.len() > 1 && alias_set_after.len() > 1 {
            return Err(bad());
        }

        Ok(Self {
            alias_set,
            is_write,
            alias_set_after,
        })
    }
}

/// Parse a `|`-separated set of single lowercase-ASCII-letter alias
/// names (e.g. `a`, `a|b`). Returns `None` for an empty string, or for
/// any part that is not exactly one lowercase ASCII letter.
fn parse_alias_set(s: &str) -> Option<Vec<char>> {
    if s.is_empty() {
        return None;
    }
    s.split('|')
        .map(|part| {
            let mut chars = part.chars();
            match (chars.next(), chars.next()) {
                (Some(c), None) if c.is_ascii_lowercase() => Some(c),
                _ => None,
            }
        })
        .collect()
}

/// Render a `|`-separated alias set back to its schema spelling. Shared
/// by both `alias_set` and `alias_set_after` so the join logic exists
/// once.
fn join_alias_set(set: &[char]) -> String {
    set.iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("|")
}

impl fmt::Display for Annotation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", join_alias_set(&self.alias_set))?;
        if self.is_write {
            write!(f, "!")?;
        }
        if !self.alias_set_after.is_empty() {
            write!(f, " -> {}", join_alias_set(&self.alias_set_after))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Annotation;

    fn round_trip(s: &str) {
        let a = Annotation::parse(s).expect("parses");
        assert_eq!(a.to_string(), s);
    }

    #[test]
    fn a_read_only_alias_has_no_write_flag() {
        let a = Annotation::parse("a").expect("parses");
        assert_eq!(a.alias_set, vec!['a']);
        assert!(!a.is_write);
        round_trip("a");
    }

    #[test]
    fn a_bang_marks_a_write() {
        let a = Annotation::parse("a!").expect("parses");
        assert!(a.is_write);
        round_trip("a!");
    }

    #[test]
    fn a_mutable_reference_may_belong_to_only_one_alias_set() {
        assert!(
            Annotation::parse("a|b!").is_err(),
            "aliasing XOR mutation: this is Rust's exclusivity rule"
        );
    }

    #[test]
    fn a_read_only_annotation_may_name_several_sets() {
        let a = Annotation::parse("a|b").expect("parses");
        assert_eq!(a.alias_set, vec!['a', 'b']);
        round_trip("a|b");
    }

    #[test]
    fn the_wildcard_escape_round_trips() {
        round_trip("a -> *");
    }

    /// Shapes counted directly from `native_functions.yaml`
    /// (`pytorch/pytorch @ 43f8f31f6a2`): `(a!)` (1863x), `(a)` (168x),
    /// `(b!)` (129x), `(g!)` (2x, the rarest lettered set that still
    /// appears), and `(a -> *)` (14x, the wildcard escape used by the
    /// split/chunk/unbind family and the only `->` shape in the corpus).
    #[test]
    fn corpus_shapes_round_trip() {
        for s in ["a!", "a", "b!", "g!", "a -> *"] {
            round_trip(s);
        }
    }

    #[test]
    fn exclusivity_error_names_the_offending_input() {
        let err = Annotation::parse("a|b!").expect_err("must reject");
        match err {
            crate::ParseError::BadAnnotation(offending) => {
                assert_eq!(offending, "a|b!");
            }
            other => panic!("expected BadAnnotation, got {other:?}"),
        }
    }

    /// `alias_set` and `alias_set_after` cannot both name more than one
    /// letter -- torchgen's own second restriction, distinct from the
    /// write-exclusivity rule. No corpus schema exercises multi-letter
    /// sets at all, so this crate's suite is the only thing that checks
    /// it: the rejection alone would pass even if the guard rejected
    /// every multi-letter after-set, so `a_single_before_set_may_pair_
    /// with_a_multi_letter_after_set` below proves the guard fires
    /// precisely, not just that it fires.
    #[test]
    fn before_and_after_sets_cannot_both_be_multi_letter() {
        let err = Annotation::parse("a|b -> c|d").expect_err("must reject");
        match err {
            crate::ParseError::BadAnnotation(offending) => {
                assert_eq!(offending, "a|b -> c|d");
            }
            other => panic!("expected BadAnnotation, got {other:?}"),
        }
    }

    #[test]
    fn a_single_before_set_may_pair_with_a_multi_letter_after_set() {
        let a = Annotation::parse("a -> b|c").expect("parses");
        assert_eq!(a.alias_set, vec!['a']);
        assert_eq!(a.alias_set_after, vec!['b', 'c']);
        round_trip("a -> b|c");
    }
}
