// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::cursor::Cursor;
use crate::error::{GgufError, Result};

const MAX_ARRAY_DEPTH: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GGUFValueType {
    Uint8,
    Int8,
    Uint16,
    Int16,
    Uint32,
    Int32,
    Float32,
    Bool,
    String,
    Array,
    Uint64,
    Int64,
    Float64,
}

impl GGUFValueType {
    fn from_u32(tag: u32) -> Option<Self> {
        usize::try_from(tag)
            .ok()
            .and_then(|index| LOOKUP.get(index).copied().flatten())
    }

    fn read(cursor: &mut Cursor<'_>) -> Result<Self> {
        let tag = cursor.u32()?;
        Self::from_u32(tag).ok_or(GgufError::InvalidValueType(tag))
    }
}

const LOOKUP: [Option<GGUFValueType>; 13] = [
    Some(GGUFValueType::Uint8),
    Some(GGUFValueType::Int8),
    Some(GGUFValueType::Uint16),
    Some(GGUFValueType::Int16),
    Some(GGUFValueType::Uint32),
    Some(GGUFValueType::Int32),
    Some(GGUFValueType::Float32),
    Some(GGUFValueType::Bool),
    Some(GGUFValueType::String),
    Some(GGUFValueType::Array),
    Some(GGUFValueType::Uint64),
    Some(GGUFValueType::Int64),
    Some(GGUFValueType::Float64),
];

/// A GGUF metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum GGUFValue {
    /// Unsigned 8-bit integer.
    Uint8(u8),
    /// Signed 8-bit integer.
    Int8(i8),
    /// Unsigned 16-bit integer.
    Uint16(u16),
    /// Signed 16-bit integer.
    Int16(i16),
    /// Unsigned 32-bit integer.
    Uint32(u32),
    /// Signed 32-bit integer.
    Int32(i32),
    /// 32-bit float.
    Float32(f32),
    /// Boolean (any non-zero byte is true).
    Bool(bool),
    /// UTF-8 string (invalid sequences replaced).
    String(String),
    /// Homogeneous array.
    Array(Vec<Self>),
    /// Unsigned 64-bit integer.
    Uint64(u64),
    /// Signed 64-bit integer.
    Int64(i64),
    /// 64-bit float.
    Float64(f64),
}

pub(crate) fn parse_typed_value(cursor: &mut Cursor<'_>) -> Result<GGUFValue> {
    let ty = GGUFValueType::read(cursor)?;
    parse_value(cursor, ty)
}

pub(crate) fn parse_value(cursor: &mut Cursor<'_>, ty: GGUFValueType) -> Result<GGUFValue> {
    parse_value_at_depth(cursor, ty, 0)
}

fn parse_value_at_depth(
    cursor: &mut Cursor<'_>,
    ty: GGUFValueType,
    depth: usize,
) -> Result<GGUFValue> {
    Ok(match ty {
        GGUFValueType::Uint8 => GGUFValue::Uint8(u8::from_le_bytes(cursor.array()?)),
        GGUFValueType::Int8 => GGUFValue::Int8(i8::from_le_bytes(cursor.array()?)),
        GGUFValueType::Uint16 => GGUFValue::Uint16(u16::from_le_bytes(cursor.array()?)),
        GGUFValueType::Int16 => GGUFValue::Int16(i16::from_le_bytes(cursor.array()?)),
        GGUFValueType::Uint32 => GGUFValue::Uint32(cursor.u32()?),
        GGUFValueType::Int32 => GGUFValue::Int32(i32::from_le_bytes(cursor.array()?)),
        GGUFValueType::Float32 => GGUFValue::Float32(f32::from_le_bytes(cursor.array()?)),
        GGUFValueType::Bool => GGUFValue::Bool(u8::from_le_bytes(cursor.array()?) != 0),
        GGUFValueType::String => GGUFValue::String(cursor.string()?),
        GGUFValueType::Array => parse_array(cursor, depth + 1)?,
        GGUFValueType::Uint64 => GGUFValue::Uint64(cursor.u64()?),
        GGUFValueType::Int64 => GGUFValue::Int64(i64::from_le_bytes(cursor.array()?)),
        GGUFValueType::Float64 => GGUFValue::Float64(f64::from_le_bytes(cursor.array()?)),
    })
}

/// `depth` is the nesting level of the array being parsed (outermost is 1).
/// Untrusted input nests arrays at about 12 bytes per level, so unbounded
/// recursion would overflow the stack.
fn parse_array(cursor: &mut Cursor<'_>, depth: usize) -> Result<GGUFValue> {
    if depth > MAX_ARRAY_DEPTH {
        return Err(GgufError::NestingTooDeep {
            limit: MAX_ARRAY_DEPTH,
        });
    }
    let item_type = GGUFValueType::read(cursor)?;
    let array_len = cursor.len()?;
    // Deliberately NOT `Vec::with_capacity(array_len)`: the length is an
    // unvalidated u64 from the file, and even clamping it to the bytes
    // remaining would still amplify by size_of::<GGUFValue>(). Growing on
    // demand costs an amortised realloc per doubling and is bounded by the
    // data actually present, since every element consumes >= 1 byte and a
    // short read ends the loop with UnexpectedEof.
    let mut items = Vec::new();
    for _ in 0..array_len {
        items.push(parse_value_at_depth(cursor, item_type, depth)?);
    }
    Ok(GGUFValue::Array(items))
}

#[cfg(test)]
#[path = "value_tests.rs"]
mod tests;
