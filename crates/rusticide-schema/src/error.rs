// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Crate-level error type.
//!
//! `ParseError` is used by every schema-grammar parser in this crate --
//! `Type` (Task 10), `Annotation` and `Argument` (Task 11), and
//! `FunctionSchema` (Task 13) all return it. It lives in its own module
//! rather than inside `ty` because it is not owned by any one of those
//! parsers.

/// Errors from parsing schema fragments.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ParseError {
    /// A base type name that is not part of the schema grammar.
    #[error("unrecognized type: {0}")]
    UnknownType(String),

    /// A list suffix (`[..]`) whose size could not be parsed as a `usize`.
    #[error("malformed list size in: {0}")]
    BadListSize(String),

    /// An alias-annotation fragment (e.g. `a!`, `a!b|c`) that does not match
    /// the annotation grammar.
    #[error("unrecognized alias annotation: {0}")]
    BadAnnotation(String),

    /// A full function-schema string that does not match the schema grammar.
    #[error("malformed schema: {0}")]
    BadSchema(String),

    /// A parsed value printed back to a string different from its input.
    #[error("round-trip mismatch: parsed {parsed}, expected {expected}")]
    RoundTrip {
        /// What `Display` produced for the parsed value.
        parsed: String,
        /// The original input string.
        expected: String,
    },
}

/// Return `value` if it prints back to exactly `input`, else a
/// [`ParseError::RoundTrip`] naming both spellings.
///
/// Shared by every parser that asserts round-tripping, so the check and
/// the error it builds exist once.
pub(crate) fn ensure_round_trip<T: core::fmt::Display>(
    value: T,
    input: &str,
) -> Result<T, ParseError> {
    let printed = value.to_string();
    if printed == input {
        Ok(value)
    } else {
        Err(ParseError::RoundTrip {
            parsed: printed,
            expected: input.to_owned(),
        })
    }
}
