// SPDX-FileCopyrightText: 2026 Rusticide contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

/// A scalar element type.
///
/// Ordering of the variants is not semantically meaningful; promotion is
/// decided by `promotion_rank`, not by variant order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    U8,
    I8,
    I16,
    I32,
    I64,
    F16,
    BF16,
    F32,
    F64,
}

impl DType {
    #[must_use]
    pub const fn size_in_bytes(self) -> usize {
        match self {
            Self::Bool | Self::U8 | Self::I8 => 1,
            Self::I16 | Self::F16 | Self::BF16 => 2,
            Self::I32 | Self::F32 => 4,
            Self::I64 | Self::F64 => 8,
        }
    }

    #[must_use]
    pub const fn is_floating_point(self) -> bool {
        matches!(self, Self::F16 | Self::BF16 | Self::F32 | Self::F64)
    }

    #[must_use]
    pub const fn is_signed(self) -> bool {
        !matches!(self, Self::Bool | Self::U8)
    }

    /// Rank within a promotion category. Higher wins.
    const fn promotion_rank(self) -> u8 {
        match self {
            Self::Bool => 0,
            Self::U8 | Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 3,
            Self::I64 => 4,
            Self::F16 | Self::BF16 => 5,
            Self::F32 => 6,
            Self::F64 => 7,
        }
    }

    /// The type both operands are cast to before an operation.
    ///
    /// Floating point always beats integral. F16 and BF16 have no
    /// ordering between them, so mixing them widens to F32 rather than
    /// silently picking one. Similarly, U8 and I8 have no ordering
    /// between them, so mixing them widens to I16.
    #[must_use]
    pub const fn promote(self, other: Self) -> Self {
        if matches!(
            (self, other),
            (Self::F16, Self::BF16) | (Self::BF16, Self::F16)
        ) {
            return Self::F32;
        }
        if matches!(
            (self, other),
            (Self::U8, Self::I8) | (Self::I8, Self::U8)
        ) {
            return Self::I16;
        }
        if self.is_floating_point() == other.is_floating_point() {
            if self.promotion_rank() >= other.promotion_rank() {
                self
            } else {
                other
            }
        } else if self.is_floating_point() {
            self
        } else {
            other
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DType;

    #[test]
    fn size_in_bytes_matches_the_scalar_width() {
        assert_eq!(DType::F32.size_in_bytes(), 4);
        assert_eq!(DType::F16.size_in_bytes(), 2);
        assert_eq!(DType::I64.size_in_bytes(), 8);
        assert_eq!(DType::Bool.size_in_bytes(), 1);
    }

    #[test]
    fn promotion_picks_the_wider_type() {
        assert_eq!(DType::I32.promote(DType::F32), DType::F32);
        assert_eq!(DType::F32.promote(DType::I32), DType::F32);
        assert_eq!(DType::I32.promote(DType::I64), DType::I64);
        assert_eq!(DType::Bool.promote(DType::I8), DType::I8);
    }

    #[test]
    fn promotion_of_half_widths_yields_the_wider_float() {
        assert_eq!(DType::F16.promote(DType::BF16), DType::F32);
        assert_eq!(DType::BF16.promote(DType::F16), DType::F32);
        assert_eq!(DType::F16.promote(DType::F32), DType::F32);
    }

    #[test]
    fn promotion_of_u8_and_i8_yields_i16() {
        assert_eq!(DType::U8.promote(DType::I8), DType::I16);
        assert_eq!(DType::I8.promote(DType::U8), DType::I16);
    }

    #[test]
    fn promotion_is_commutative_for_every_pair() {
        let all_types = [
            DType::Bool,
            DType::U8,
            DType::I8,
            DType::I16,
            DType::I32,
            DType::I64,
            DType::F16,
            DType::BF16,
            DType::F32,
            DType::F64,
        ];

        for a in all_types {
            for b in all_types {
                assert_eq!(a.promote(b), b.promote(a));
            }
        }
    }
}
