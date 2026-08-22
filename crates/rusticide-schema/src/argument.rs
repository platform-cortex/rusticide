// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Argument` and `Arguments`: the argument-list grammar of an operator
//! schema.
//!
//! `Return` lives in the sibling `returns` module (`return` is a Rust
//! keyword, so it cannot be a module name) but shares this module's
//! `split_annotation`/`write_annotated_type` helpers, since a return is
//! the same `Type[(annotation)][ name]` shape without a default.

use core::fmt;

use crate::{Annotation, ParseError, Type};

/// One argument in an operator schema: `Type[(annotation)] name[=default]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Argument {
    /// The argument's name.
    pub name: String,
    /// The argument's type.
    pub ty: Type,
    /// The alias annotation attached to the type, if any.
    pub annotation: Option<Annotation>,
    /// The default value, kept as opaque, verbatim schema text (e.g.
    /// `None`, `Mean`, `[-2,-1]`) rather than parsed to a typed value.
    pub default: Option<String>,
}

impl Argument {
    /// Parse an argument. The type and name are split on the last space
    /// outside of the default (so a defaulted string containing a space
    /// still works, since the default is split off first); the
    /// annotation is split off the type by its parenthesised suffix.
    ///
    /// # Errors
    ///
    /// Returns [`ParseError::BadSchema`] if `s` has no type/name
    /// separator. Propagates [`ParseError`] from the nested [`Type`] and
    /// [`Annotation`] parses.
    pub fn parse(s: &str) -> Result<Self, ParseError> {
        let bad = || ParseError::BadSchema(s.to_owned());
        let (type_and_name, default) = match s.split_once('=') {
            Some((lhs, rhs)) => (lhs, Some(rhs.to_owned())),
            None => (s, None),
        };
        let (type_str, name) = type_and_name.rsplit_once(' ').ok_or_else(bad)?;
        let (ty, annotation) = split_annotation(type_str)?;
        Ok(Self {
            name: name.to_owned(),
            ty,
            annotation,
            default,
        })
    }

    /// Whether this argument's annotation marks it as written-through.
    #[must_use]
    pub fn is_write(&self) -> bool {
        self.annotation.as_ref().is_some_and(|a| a.is_write)
    }
}

/// Split `Tensor(a!)` into its type and its annotation. The annotation
/// binds to the base type, before any `?`/`[]` suffix, so it is spliced
/// out of the middle of the type string before the remainder is parsed
/// as a `Type`.
///
/// Shared with the sibling `returns` module: a return is `Argument`
/// without a name-and-default split, but the same annotation splice.
pub(crate) fn split_annotation(s: &str) -> Result<(Type, Option<Annotation>), ParseError> {
    let bad = || ParseError::BadSchema(s.to_owned());
    let Some(open) = s.find('(') else {
        return Ok((Type::parse(s)?, None));
    };
    let close = s.rfind(')').ok_or_else(bad)?;
    let ann = Annotation::parse(&s[open + 1..close])?;
    let mut ty_str = String::with_capacity(s.len());
    ty_str.push_str(&s[..open]);
    ty_str.push_str(&s[close + 1..]);
    Ok((Type::parse(&ty_str)?, Some(ann)))
}

/// Splice an annotation into a printed type's base, before its first
/// `?`/`[` suffix character: `Tensor(a!)`, `Tensor(a!)[]`, never
/// `Tensor[](a!)`. Shared with the sibling `returns` module.
pub(crate) fn write_annotated_type(
    f: &mut fmt::Formatter<'_>,
    printed: &str,
    annotation: Option<&Annotation>,
) -> fmt::Result {
    let Some(a) = annotation else {
        return write!(f, "{printed}");
    };
    let base_len = printed.find(['?', '[']).unwrap_or(printed.len());
    write!(f, "{}({}){}", &printed[..base_len], a, &printed[base_len..])
}

impl fmt::Display for Argument {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_annotated_type(f, &self.ty.to_string(), self.annotation.as_ref())?;
        write!(f, " {}", self.name)?;
        if let Some(d) = &self.default {
            write!(f, "={d}")?;
        }
        Ok(())
    }
}

/// The argument list of a schema, split at the `*` kwarg-only marker.
///
/// Out arguments (annotated as writes) are further separated from the
/// other kwarg-only arguments so `FunctionSchema` (Task 13) can treat
/// them distinctly, as torchgen does.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Arguments {
    /// Arguments before the `*` marker (or all arguments, if there is
    /// none).
    pub positional: Vec<Argument>,
    /// Arguments after the `*` marker that are not out arguments.
    pub kwarg_only: Vec<Argument>,
    /// Arguments after the `*` marker that are annotated as writes.
    pub out: Vec<Argument>,
}

impl Arguments {
    /// Parse a comma-separated argument list.
    ///
    /// # Errors
    ///
    /// Propagates [`ParseError`] from any individual [`Argument::parse`].
    pub fn parse(s: &str) -> Result<Self, ParseError> {
        let mut out = Self::default();
        if s.trim().is_empty() {
            return Ok(out);
        }
        let mut after_star = false;
        for part in split_top_level(s) {
            let part = part.trim();
            if part == "*" {
                after_star = true;
                continue;
            }
            let arg = Argument::parse(part)?;
            if !after_star {
                out.positional.push(arg);
            } else if arg.is_write() {
                out.out.push(arg);
            } else {
                out.kwarg_only.push(arg);
            }
        }
        Ok(out)
    }
}

/// Split on commas that are not inside `[..]`/`(..)`, so a default like
/// `int[2] dim=[-2,-1]` stays one argument. Naively splitting on every
/// comma severs it into `int[2] dim=[-2` and `-1]`.
fn split_top_level(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    for (i, c) in s.char_indices() {
        match c {
            '[' | '(' => depth += 1,
            ']' | ')' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&s[start..]);
    parts
}

impl fmt::Display for Arguments {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts: Vec<String> = self.positional.iter().map(ToString::to_string).collect();
        if !self.kwarg_only.is_empty() || !self.out.is_empty() {
            parts.push("*".to_owned());
            parts.extend(self.kwarg_only.iter().map(ToString::to_string));
            parts.extend(self.out.iter().map(ToString::to_string));
        }
        write!(f, "{}", parts.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::{Argument, Arguments};

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
}
