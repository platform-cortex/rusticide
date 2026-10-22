// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{Error, Result, Shape, Tensor};

/// Legacy eager CPU operator (SP0 port of the deleted rusticide-tensor).
/// Bypasses the dispatcher. Replaced by dispatched operators in SP2; do not
/// add new operators here.
pub(super) fn reshape(tensor: &Tensor, dims: &[usize]) -> Result<Tensor> {
    let shape = Shape::new(dims);
    if shape.numel() != tensor.numel() {
        return Err(Error::ElementCountMismatch {
            expected: tensor.numel(),
            got: shape.numel(),
        });
    }
    tensor.copy_with_shape(shape)
}

/// Legacy eager CPU operator (SP0 port of the deleted rusticide-tensor).
/// Bypasses the dispatcher. Replaced by dispatched operators in SP2; do not
/// add new operators here.
pub(super) fn contiguous(tensor: &Tensor) -> Tensor {
    tensor.clone()
}

#[cfg(test)]
mod tests {
    use crate::{EagerOps, Error, Tensor};

    fn matrix() -> Tensor {
        Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("valid data")
    }

    #[test]
    fn reshape_keeps_row_major_order() {
        let r = matrix().reshape(&[3, 2]).expect("same count");
        assert_eq!(r.shape().dims(), &[3, 2]);
        assert_eq!(
            r.to_vec::<f32>().expect("f32"),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
        assert_eq!(r.get::<f32>(&[2, 0]).expect("in bounds"), 5.0);
    }

    #[test]
    fn reshape_does_not_alias_its_source() {
        let source = matrix();
        let r = source.reshape(&[6]).expect("same count");
        r.set(&[0], 9.0f32).expect("in bounds");
        assert_eq!(source.get::<f32>(&[0, 0]).expect("in bounds"), 1.0);
        assert!(!source.shares_storage_with(&r));
    }

    #[test]
    fn reshape_rejects_a_different_element_count() {
        let err = matrix().reshape(&[4, 2]).expect_err("6 != 8");
        assert!(matches!(
            err,
            Error::ElementCountMismatch {
                expected: 6,
                got: 8
            }
        ));
    }

    #[test]
    fn contiguous_aliases_its_source() {
        let source = matrix();
        let c = source.contiguous();
        assert!(source.shares_storage_with(&c));
        assert_eq!(c.shape().dims(), &[2, 3]);
    }
}
