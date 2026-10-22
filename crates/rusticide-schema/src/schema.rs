// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `FunctionSchema`: a complete operator signature.

use core::fmt;

use crate::error::ensure_round_trip;
use crate::{Argument, Arguments, OperatorName, ParseError, Return};

/// A complete operator signature: `name(arguments) -> returns`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionSchema {
    /// The operator's name and overload tag.
    pub name: OperatorName,
    /// The operator's arguments.
    pub arguments: Arguments,
    /// The operator's return values; empty for `()`.
    pub returns: Vec<Return>,
}

impl FunctionSchema {
    /// Parse a full schema such as `add.Tensor(Tensor self, Tensor other)
    /// -> Tensor`, requiring that it prints back byte-exactly.
    ///
    /// # Errors
    ///
    /// Returns [`ParseError::BadSchema`] if the `(` or the `) -> `
    /// separator is missing, propagates errors from [`Arguments::parse`]
    /// and [`Return::parse`], and returns [`ParseError::RoundTrip`] if the
    /// parsed value does not print back to `s`.
    pub fn parse(s: &str) -> Result<Self, ParseError> {
        let bad = || ParseError::BadSchema(s.to_owned());
        let open = s.find('(').ok_or_else(bad)?;
        let arrow = s.rfind(") -> ").ok_or_else(bad)?;
        let arguments_str = s.get(open + 1..arrow).ok_or_else(bad)?;
        let schema = Self {
            name: OperatorName::parse(&s[..open]),
            arguments: Arguments::parse(arguments_str)?,
            returns: parse_returns(&s[arrow + ") -> ".len()..])?,
        };
        ensure_round_trip(schema, s)
    }

    /// Whether any argument is annotated as written.
    #[must_use]
    pub fn is_mutable(&self) -> bool {
        self.arguments
            .positional
            .iter()
            .chain(&self.arguments.kwarg_only)
            .chain(&self.arguments.out)
            .any(Argument::is_write)
    }
}

/// Parse the text after `->`: `()`, a single return, or a parenthesised
/// comma-separated tuple of returns.
fn parse_returns(s: &str) -> Result<Vec<Return>, ParseError> {
    if s == "()" {
        return Ok(Vec::new());
    }
    match s.strip_prefix('(').and_then(|t| t.strip_suffix(')')) {
        Some(inner) => inner.split(',').map(|r| Return::parse(r.trim())).collect(),
        None => Ok(vec![Return::parse(s)?]),
    }
}

impl fmt::Display for FunctionSchema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({}) -> ", self.name, self.arguments)?;
        match self.returns.as_slice() {
            [] => write!(f, "()"),
            [single] => write!(f, "{single}"),
            many => {
                let joined: Vec<String> = many.iter().map(ToString::to_string).collect();
                write!(f, "({})", joined.join(", "))
            }
        }
    }
}

#[cfg(test)]
mod tests;
