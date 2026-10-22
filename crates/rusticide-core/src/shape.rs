// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{Error, Result};

/// The extent of each dimension of a tensor.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    #[must_use]
    pub fn new(dims: &[usize]) -> Self {
        Self {
            dims: dims.to_vec(),
        }
    }

    #[must_use]
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    #[must_use]
    pub const fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Total element count. A zero-dimensional shape holds one element.
    ///
    /// A shape with any zero extent holds zero elements, even if the other
    /// extents would overflow. Otherwise a product that overflows `usize`
    /// saturates to `usize::MAX`, which no allocation or slice can satisfy.
    #[must_use]
    pub fn numel(&self) -> usize {
        if self.dims.contains(&0) {
            return 0;
        }
        self.dims
            .iter()
            .try_fold(1usize, |count, &extent| count.checked_mul(extent))
            .unwrap_or(usize::MAX)
    }

    /// `NumPy` broadcasting: align from the right, pad the shorter shape
    /// with implicit leading ones, and stretch any extent of 1.
    ///
    /// # Errors
    ///
    /// Returns `Error::NotBroadcastable` if the shapes have incompatible
    /// dimensions that cannot be broadcast together.
    pub fn broadcast_with(&self, other: &Self) -> Result<Self> {
        let ndim = self.ndim().max(other.ndim());
        let mut dims = Vec::with_capacity(ndim);
        for i in 0..ndim {
            let a = self.dim_from_right(i);
            let b = other.dim_from_right(i);
            let d = match (a, b) {
                (x, y) if x == y => x,
                (1, y) => y,
                (x, 1) => x,
                _ => {
                    return Err(Error::NotBroadcastable {
                        lhs: self.dims.clone(),
                        rhs: other.dims.clone(),
                    });
                }
            };
            dims.push(d);
        }
        dims.reverse();
        Ok(Self { dims })
    }

    /// Extent `i` positions from the right, or 1 past the left edge.
    fn dim_from_right(&self, i: usize) -> usize {
        if i < self.dims.len() {
            self.dims[self.dims.len() - 1 - i]
        } else {
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Shape;

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn numel_saturates_on_overflow() {
        assert_eq!(Shape::new(&[1 << 32, 1 << 32]).numel(), usize::MAX);
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn numel_is_zero_when_any_dim_is_zero_even_if_the_rest_overflow() {
        assert_eq!(Shape::new(&[1 << 32, 1 << 32, 0]).numel(), 0);
    }

    #[test]
    fn numel_is_the_product_of_dims() {
        assert_eq!(Shape::new(&[2, 3, 4]).numel(), 24);
    }

    #[test]
    fn a_scalar_has_zero_dims_and_one_element() {
        let s = Shape::new(&[]);
        assert_eq!(s.ndim(), 0);
        assert_eq!(s.numel(), 1);
    }

    #[test]
    fn a_zero_dim_makes_numel_zero() {
        assert_eq!(Shape::new(&[2, 0, 4]).numel(), 0);
    }

    #[test]
    fn broadcast_stretches_ones_and_pads_on_the_left() {
        let a = Shape::new(&[3, 1, 5]);
        let b = Shape::new(&[4, 5]);
        let r = a.broadcast_with(&b).expect("shapes broadcast");
        assert_eq!(r.dims(), &[3, 4, 5]);
    }

    #[test]
    fn broadcast_rejects_incompatible_dims() {
        let a = Shape::new(&[3, 2]);
        let b = Shape::new(&[3, 4]);
        assert!(a.broadcast_with(&b).is_err());
    }
}
