// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Type`: a schema type, and `BaseTy`, its atomic base-type vocabulary.

use core::fmt;

use crate::ParseError;

/// An atomic type in the operator schema language.
///
/// This is the complete vocabulary of base types used across `PyTorch`'s
/// `native_functions.yaml`. Adding a variant that the corpus never uses
/// would go unnoticed; dropping one that it does use fails loudly the
/// first time `Type::parse` sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BaseTy {
    /// `bool`.
    Bool,
    /// `Device`.
    Device,
    /// `DeviceIndex`.
    DeviceIndex,
    /// `DimVector`.
    DimVector,
    /// `float`.
    Float,
    /// `Generator`.
    Generator,
    /// `GraphModule`.
    GraphModule,
    /// `int`.
    Int,
    /// `Layout`.
    Layout,
    /// `MemoryFormat`.
    MemoryFormat,
    /// `QScheme`.
    QScheme,
    /// `Scalar`.
    Scalar,
    /// `ScalarType`.
    ScalarType,
    /// `Storage`.
    Storage,
    /// `str`.
    Str,
    /// `Stream`.
    Stream,
    /// `SymBool`.
    SymBool,
    /// `SymInt`.
    SymInt,
    /// `Tensor`.
    Tensor,
}

impl BaseTy {
    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "bool" => Self::Bool,
            "Device" => Self::Device,
            "DeviceIndex" => Self::DeviceIndex,
            "DimVector" => Self::DimVector,
            "float" => Self::Float,
            "Generator" => Self::Generator,
            "GraphModule" => Self::GraphModule,
            "int" => Self::Int,
            "Layout" => Self::Layout,
            "MemoryFormat" => Self::MemoryFormat,
            "QScheme" => Self::QScheme,
            "Scalar" => Self::Scalar,
            "ScalarType" => Self::ScalarType,
            "Storage" => Self::Storage,
            "str" => Self::Str,
            "Stream" => Self::Stream,
            "SymBool" => Self::SymBool,
            "SymInt" => Self::SymInt,
            "Tensor" => Self::Tensor,
            _ => return None,
        })
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::Device => "Device",
            Self::DeviceIndex => "DeviceIndex",
            Self::DimVector => "DimVector",
            Self::Float => "float",
            Self::Generator => "Generator",
            Self::GraphModule => "GraphModule",
            Self::Int => "int",
            Self::Layout => "Layout",
            Self::MemoryFormat => "MemoryFormat",
            Self::QScheme => "QScheme",
            Self::Scalar => "Scalar",
            Self::ScalarType => "ScalarType",
            Self::Storage => "Storage",
            Self::Str => "str",
            Self::Stream => "Stream",
            Self::SymBool => "SymBool",
            Self::SymInt => "SymInt",
            Self::Tensor => "Tensor",
        }
    }
}

impl fmt::Display for BaseTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A schema type: a base type, optionally wrapped in `?` and `[]`.
///
/// The two wrappers nest, and nesting order carries meaning: `int[]?` is an
/// optional list of int (`Optional(List(int))`), while `Tensor?[]` is a list
/// of optional tensor (`List(Optional(Tensor))`). `parse` always strips a
/// trailing `?` before a trailing `[..]`, so the outermost suffix in the
/// source string becomes the outermost wrapper in the tree.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    /// An atomic base type.
    Base(BaseTy),
    /// `<inner>?`.
    Optional(Box<Self>),
    /// `<elem>[]` (size `None`) or `<elem>[N]` (size `Some(N)`).
    List {
        /// The element type.
        elem: Box<Self>,
        /// The fixed list length, if the source specified one.
        size: Option<usize>,
    },
}

impl Type {
    /// Parse a type, asserting it round-trips.
    ///
    /// The assertion is the point: a type that cannot be printed back
    /// exactly is a type whose representation lost information.
    ///
    /// # Errors
    ///
    /// Returns [`ParseError::UnknownType`] for a base type outside the
    /// grammar's vocabulary, [`ParseError::BadListSize`] for a `[..]`
    /// suffix whose contents are not empty and not a valid `usize`, and
    /// [`ParseError::RoundTrip`] if the parsed value fails to print back
    /// to exactly `s` (a bug in this parser, not in the input).
    pub fn parse(s: &str) -> Result<Self, ParseError> {
        let t = Self::parse_inner(s)?;
        let printed = t.to_string();
        if printed == s {
            Ok(t)
        } else {
            Err(ParseError::RoundTrip {
                parsed: printed,
                expected: s.to_owned(),
            })
        }
    }

    fn parse_inner(s: &str) -> Result<Self, ParseError> {
        if let Some(inner) = s.strip_suffix('?') {
            return Ok(Self::Optional(Box::new(Self::parse_inner(inner)?)));
        }
        if let Some(open) = s.rfind('[') {
            if let Some(rest) = s.strip_suffix(']') {
                let size_str = &rest[open + 1..];
                let size = if size_str.is_empty() {
                    None
                } else {
                    Some(
                        size_str
                            .parse::<usize>()
                            .map_err(|_| ParseError::BadListSize(s.to_owned()))?,
                    )
                };
                let elem = Self::parse_inner(&s[..open])?;
                return Ok(Self::List {
                    elem: Box::new(elem),
                    size,
                });
            }
        }
        BaseTy::parse(s)
            .map(Self::Base)
            .ok_or_else(|| ParseError::UnknownType(s.to_owned()))
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Base(b) => write!(f, "{b}"),
            Self::Optional(inner) => write!(f, "{inner}?"),
            Self::List { elem, size: None } => write!(f, "{elem}[]"),
            Self::List {
                elem,
                size: Some(n),
            } => write!(f, "{elem}[{n}]"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BaseTy, Type};

    fn round_trip(s: &str) {
        let t = Type::parse(s).expect("parses");
        assert_eq!(t.to_string(), s, "round-trip must be byte exact");
    }

    #[test]
    fn base_types_parse() {
        assert_eq!(Type::parse("Tensor"), Ok(Type::Base(BaseTy::Tensor)));
        assert_eq!(Type::parse("int"), Ok(Type::Base(BaseTy::Int)));
        assert_eq!(Type::parse("SymInt"), Ok(Type::Base(BaseTy::SymInt)));
    }

    #[test]
    fn optional_and_list_nest() {
        round_trip("Tensor?");
        round_trip("int[]");
        round_trip("int[2]");
        round_trip("Tensor?[]");
        round_trip("int[]?");
    }

    #[test]
    fn unknown_base_types_are_rejected() {
        assert!(Type::parse("Frobnicator").is_err());
    }

    #[test]
    fn every_shape_round_trips() {
        for s in ["Tensor", "float", "bool[3]", "ScalarType?", "str"] {
            round_trip(s);
        }
    }

    /// Verbatim strings pulled from `PyTorch`'s `native_functions.yaml`. Every
    /// base type plus both nesting orders of `?` and `[]` must round-trip
    /// byte-exact, or Task 13's 2,585-schema corpus test would fail on real
    /// input.
    #[test]
    fn corpus_samples_round_trip() {
        for s in [
            "Tensor",
            "Tensor?",
            "Tensor[]",
            "int",
            "int[]",
            "int[1]",
            "int[]?",
            "SymInt",
            "SymInt[]",
            "SymInt[1]",
            "float",
            "float[]",
            "bool",
            "bool[3]",
            "str",
            "str?",
            "Scalar",
            "Scalar?",
            "ScalarType?",
            "Layout?",
            "Device?",
            "MemoryFormat?",
            "Generator?",
        ] {
            round_trip(s);
        }
    }
}
