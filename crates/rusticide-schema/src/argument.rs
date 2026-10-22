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
    let close = s.rfind(')').filter(|&close| open < close).ok_or_else(bad)?;
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
mod tests;
