// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Return`: an operator schema's return-value grammar.
//!
//! Named `returns` rather than `return` because `return` is a Rust
//! keyword and cannot be a module name. `Return` reuses the sibling
//! `argument` module's `split_annotation`/`write_annotated_type`
//! helpers: a return is `Type[(annotation)][ name]`, the same shape as
//! an argument minus the `=default` clause.

use core::fmt;

use crate::argument::{split_annotation, write_annotated_type};
use crate::{Annotation, ParseError, Type};

/// One return value of an operator schema: `Type[(annotation)][ name]`.
/// Unlike [`crate::Argument`], the name is optional -- most returns are
/// bare types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Return {
    /// The return's name, if the schema gives one.
    pub name: Option<String>,
    /// The return's type.
    pub ty: Type,
    /// The alias annotation attached to the type, if any.
    pub annotation: Option<Annotation>,
}

impl Return {
    /// Parse a single return. A return has no `=default` clause; a
    /// trailing space-separated word after the type is its name.
    ///
    /// # Errors
    ///
    /// Propagates [`ParseError`] from the nested [`Type`] and
    /// [`Annotation`] parses.
    pub fn parse(s: &str) -> Result<Self, ParseError> {
        let (type_str, name) = match s.rsplit_once(' ') {
            Some((t, n)) => (t, Some(n.to_owned())),
            None => (s, None),
        };
        let (ty, annotation) = split_annotation(type_str)?;
        Ok(Self {
            name,
            ty,
            annotation,
        })
    }
}

impl fmt::Display for Return {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_annotated_type(f, &self.ty.to_string(), self.annotation.as_ref())?;
        if let Some(n) = &self.name {
            write!(f, " {n}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Return;

    #[test]
    fn a_bare_return_has_no_name_and_round_trips() {
        let r = Return::parse("Tensor").expect("parses");
        assert!(r.name.is_none());
        assert!(r.annotation.is_none());
        assert_eq!(r.to_string(), "Tensor");
    }

    /// `add.out`'s single return, taken from `native_functions.yaml`
    /// (`pytorch/pytorch @ 43f8f31f6a2`): an annotated, named return.
    #[test]
    fn a_named_annotated_return_round_trips() {
        let r = Return::parse("Tensor(a!) out").expect("parses");
        assert_eq!(r.name.as_deref(), Some("out"));
        let ann = r.annotation.as_ref().expect("has annotation");
        assert!(ann.is_write);
        assert_eq!(r.to_string(), "Tensor(a!) out");
    }
}
