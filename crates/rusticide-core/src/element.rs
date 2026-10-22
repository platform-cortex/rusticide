// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::DType;
use half::f16;

mod sealed {
    #[expect(unnameable_types, reason = "sealed-trait pattern")]
    pub trait Sealed {
        fn read_ne(bytes: &[u8]) -> Self;
        fn write_ne(self, out: &mut [u8]);
    }
}

/// A Rust scalar type that maps one-to-one onto a [`DType`].
///
/// Sealed: only `f32` and `half::f16` implement it. Elements are stored
/// native-endian.
pub trait Element: Copy + Send + Sync + 'static + sealed::Sealed {
    /// The dtype this scalar type is stored as.
    const DTYPE: DType;
}

macro_rules! impl_element {
    ($type:ty, $dtype:expr, $size:literal) => {
        impl sealed::Sealed for $type {
            fn read_ne(bytes: &[u8]) -> Self {
                let mut raw = [0u8; $size];
                raw.copy_from_slice(bytes);
                Self::from_ne_bytes(raw)
            }

            fn write_ne(self, out: &mut [u8]) {
                out.copy_from_slice(&self.to_ne_bytes());
            }
        }

        impl Element for $type {
            const DTYPE: DType = $dtype;
        }
    };
}

impl_element!(f32, DType::F32, 4);
impl_element!(f16, DType::F16, 2);

#[cfg(test)]
mod tests {
    use super::Element;
    use crate::DType;
    use half::f16;

    #[test]
    fn f32_dtype_is_f32() {
        assert_eq!(<f32 as Element>::DTYPE, DType::F32);
    }

    #[test]
    fn f16_dtype_is_f16() {
        assert_eq!(<f16 as Element>::DTYPE, DType::F16);
    }

    #[test]
    fn rust_size_matches_dtype_size() {
        assert_eq!(size_of::<f32>(), <f32 as Element>::DTYPE.size_in_bytes());
        assert_eq!(size_of::<f16>(), <f16 as Element>::DTYPE.size_in_bytes());
    }

    #[test]
    fn f32_round_trips_through_native_bytes() {
        use super::sealed::Sealed;
        let mut buffer = [0u8; 4];
        1.5f32.write_ne(&mut buffer);
        assert_eq!(buffer, 1.5f32.to_ne_bytes());
        assert!((f32::read_ne(&buffer) - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn f16_round_trips_through_native_bytes() {
        use super::sealed::Sealed;
        let value = f16::from_f32(-2.0);
        let mut buffer = [0u8; 2];
        value.write_ne(&mut buffer);
        assert_eq!(buffer, value.to_ne_bytes());
        assert_eq!(f16::read_ne(&buffer), value);
    }
}
