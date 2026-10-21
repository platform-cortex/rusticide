// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::Shape;

/// The element step along each dimension.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Strides {
    strides: Vec<usize>,
}

impl Strides {
    #[must_use]
    pub fn new(strides: &[usize]) -> Self {
        Self {
            strides: strides.to_vec(),
        }
    }

    #[must_use]
    pub fn as_slice(&self) -> &[usize] {
        &self.strides
    }

    /// Row-major strides: the last dimension steps by one.
    #[must_use]
    pub fn contiguous_for(shape: &Shape) -> Self {
        let dims = shape.dims();
        let mut strides = vec![1; dims.len()];
        for i in (0..dims.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        Self { strides }
    }

    #[must_use]
    pub fn is_contiguous(&self, shape: &Shape) -> bool {
        self.strides == Self::contiguous_for(shape).strides
    }
}

#[cfg(test)]
mod tests {
    use super::Strides;
    use crate::Shape;

    #[test]
    fn contiguous_strides_are_row_major() {
        let shape = Shape::new(&[2, 3, 4]);
        assert_eq!(Strides::contiguous_for(&shape).as_slice(), &[12, 4, 1]);
    }

    #[test]
    fn contiguous_strides_recognise_themselves() {
        let shape = Shape::new(&[2, 3, 4]);
        let strides = Strides::contiguous_for(&shape);
        assert!(strides.is_contiguous(&shape));
    }

    #[test]
    fn permuted_strides_are_not_contiguous() {
        let shape = Shape::new(&[2, 3]);
        let strides = Strides::new(&[1, 2]);
        assert!(!strides.is_contiguous(&shape));
    }
}
