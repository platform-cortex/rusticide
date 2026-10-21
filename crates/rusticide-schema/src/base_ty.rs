// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `BaseTy`: the atomic base-type vocabulary of the operator schema
//! language.

use core::fmt;

/// An atomic type in the operator schema language: torchgen's complete
/// `BaseTy` vocabulary.
///
/// `DimVector` and `GraphModule` see zero use in the current
/// `native_functions.yaml` corpus but are genuine upstream members, kept
/// regardless; `BASE_TY_TABLE` gives each variant's spelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BaseTy {
    Bool,
    Device,
    DeviceIndex,
    DimVector,
    Float,
    Generator,
    GraphModule,
    Int,
    Layout,
    MemoryFormat,
    QScheme,
    Scalar,
    ScalarType,
    Storage,
    Str,
    Stream,
    SymBool,
    SymInt,
    Tensor,
}

/// Sole source of `BaseTy`'s string spelling; `parse` and `as_str` both
/// read it instead of each keeping their own copy, so a typo can no
/// longer be made in one direction without the other.
const BASE_TY_TABLE: [(&str, BaseTy); 19] = [
    ("bool", BaseTy::Bool),
    ("Device", BaseTy::Device),
    ("DeviceIndex", BaseTy::DeviceIndex),
    ("DimVector", BaseTy::DimVector),
    ("float", BaseTy::Float),
    ("Generator", BaseTy::Generator),
    ("GraphModule", BaseTy::GraphModule),
    ("int", BaseTy::Int),
    ("Layout", BaseTy::Layout),
    ("MemoryFormat", BaseTy::MemoryFormat),
    ("QScheme", BaseTy::QScheme),
    ("Scalar", BaseTy::Scalar),
    ("ScalarType", BaseTy::ScalarType),
    ("Storage", BaseTy::Storage),
    ("str", BaseTy::Str),
    ("Stream", BaseTy::Stream),
    ("SymBool", BaseTy::SymBool),
    ("SymInt", BaseTy::SymInt),
    ("Tensor", BaseTy::Tensor),
];

/// Fails to compile if a `BaseTy` variant is added without a matching
/// `BASE_TY_TABLE` entry. The match is exhaustive over the type, so rustc
/// rejects it (E0004) the moment a variant is missing here -- the
/// guarantee a runtime length check cannot give, since a forgotten table
/// entry leaves `BASE_TY_TABLE.len()` unchanged.
#[allow(dead_code)]
const fn assert_every_base_ty_is_in_the_table(t: BaseTy) {
    match t {
        BaseTy::Bool
        | BaseTy::Device
        | BaseTy::DeviceIndex
        | BaseTy::DimVector
        | BaseTy::Float
        | BaseTy::Generator
        | BaseTy::GraphModule
        | BaseTy::Int
        | BaseTy::Layout
        | BaseTy::MemoryFormat
        | BaseTy::QScheme
        | BaseTy::Scalar
        | BaseTy::ScalarType
        | BaseTy::Storage
        | BaseTy::Str
        | BaseTy::Stream
        | BaseTy::SymBool
        | BaseTy::SymInt
        | BaseTy::Tensor => (),
    }
}

impl BaseTy {
    pub(crate) fn parse(s: &str) -> Option<Self> {
        BASE_TY_TABLE
            .iter()
            .find_map(|&(name, ty)| (name == s).then_some(ty))
    }

    /// `assert_every_base_ty_is_in_the_table` makes the fallback below
    /// unreachable in practice; it exists so this stays panic-free rather
    /// than relying on that invariant.
    pub(crate) fn as_str(self) -> &'static str {
        BASE_TY_TABLE
            .iter()
            .find_map(|&(name, ty)| (ty == self).then_some(name))
            .unwrap_or("<BaseTy missing from BASE_TY_TABLE>")
    }
}

impl fmt::Display for BaseTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::BaseTy;

    /// Renamed from `base_types_parse` (which called `Type::parse`) once
    /// direct coverage of `BaseTy::parse` closed the gap: as originally
    /// written, `BaseTy::parse` had zero direct test coverage, only
    /// transitive coverage through `Type::parse`'s delegation.
    #[test]
    fn parse_recognizes_known_base_types() {
        assert_eq!(BaseTy::parse("Tensor"), Some(BaseTy::Tensor));
        assert_eq!(BaseTy::parse("int"), Some(BaseTy::Int));
        assert_eq!(BaseTy::parse("SymInt"), Some(BaseTy::SymInt));
    }

    #[test]
    fn parse_rejects_unknown_base_types() {
        assert_eq!(BaseTy::parse("Frobnicator"), None);
    }

    /// The guard's work is done by rustc; calling it here only proves it
    /// accepts every tabled variant without panicking.
    #[test]
    fn the_exhaustiveness_guard_accepts_every_tabled_base_ty() {
        for (_, ty) in super::BASE_TY_TABLE {
            super::assert_every_base_ty_is_in_the_table(ty);
        }
    }
}
