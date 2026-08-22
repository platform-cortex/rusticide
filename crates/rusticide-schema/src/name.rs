// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `OperatorName`: an operator's base name and optional overload tag.

use core::fmt;

/// An operator's name and optional overload tag: `add.Tensor`.
///
/// This grammar cannot fail to parse -- any string is a valid name with
/// or without an overload -- so, unlike the other parsers in this
/// crate, `parse` is infallible and returns `Self` rather than
/// `Result<Self, ParseError>`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OperatorName {
    /// The base operator name, e.g. `add`.
    pub name: String,
    /// The overload tag after the first `.`, e.g. `Tensor`.
    pub overload: Option<String>,
}

impl OperatorName {
    /// Split `s` on its first `.` into a name and an overload tag.
    #[must_use]
    pub fn parse(s: &str) -> Self {
        match s.split_once('.') {
            Some((n, o)) => Self {
                name: n.to_owned(),
                overload: Some(o.to_owned()),
            },
            None => Self {
                name: s.to_owned(),
                overload: None,
            },
        }
    }

    /// A trailing `_` marks an in-place operator (`add_`); a trailing
    /// `__` is a dunder name (`__and__`) rather than an in-place marker.
    #[must_use]
    pub fn is_inplace(&self) -> bool {
        self.name.ends_with('_') && !self.name.ends_with("__")
    }
}

impl fmt::Display for OperatorName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.overload {
            Some(o) => write!(f, "{}.{o}", self.name),
            None => write!(f, "{}", self.name),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::OperatorName;

    #[test]
    fn a_bare_name_has_no_overload() {
        let n = OperatorName::parse("add");
        assert_eq!(n.name, "add");
        assert!(n.overload.is_none());
        assert_eq!(n.to_string(), "add");
    }

    #[test]
    fn a_dotted_name_splits_off_its_overload() {
        let n = OperatorName::parse("add.Tensor");
        assert_eq!(n.name, "add");
        assert_eq!(n.overload.as_deref(), Some("Tensor"));
        assert_eq!(n.to_string(), "add.Tensor");
    }

    #[test]
    fn a_trailing_underscore_marks_an_inplace_operator() {
        assert!(OperatorName::parse("add_").is_inplace());
        assert!(!OperatorName::parse("add").is_inplace());
    }

    /// `__and__` and friends are dunder methods, not in-place operators:
    /// the trailing-underscore check must not fire on a double underscore.
    #[test]
    fn a_double_trailing_underscore_is_not_inplace() {
        assert!(!OperatorName::parse("add__").is_inplace());
    }
}
