// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{filled_f32, require_f32};
use crate::{Result, Tensor};

/// Legacy eager CPU operator (SP0 port of the deleted rusticide-tensor).
/// Bypasses the dispatcher. Replaced by dispatched operators in SP2; do not
/// add new operators here.
pub(super) fn add(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    require_f32(lhs)?;
    require_f32(rhs)?;
    let out = lhs.shape().broadcast_with(rhs.shape())?;
    let left = lhs.to_vec::<f32>()?;
    let right = rhs.to_vec::<f32>()?;
    filled_f32(out.dims(), |flat| {
        left[source_index(flat, out.dims(), lhs.shape().dims())]
            + right[source_index(flat, out.dims(), rhs.shape().dims())]
    })
}

/// Map a flat row-major index of the broadcast output to the flat index of
/// the source element it reads: dims are right-aligned and extent-1 dims
/// are pinned to 0.
fn source_index(flat: usize, out: &[usize], src: &[usize]) -> usize {
    let mut remaining = flat;
    let mut stride = 1;
    let mut index = 0;
    for (axis, extent) in out.iter().enumerate().rev() {
        let coordinate = remaining % extent;
        remaining /= extent;
        let offset = out.len() - axis;
        if offset <= src.len() {
            let source_extent = src[src.len() - offset];
            if source_extent != 1 {
                index += coordinate * stride;
            }
            stride *= source_extent;
        }
    }
    index
}

#[cfg(test)]
mod tests {
    use crate::{DType, EagerOps, Error, Tensor};
    use half::f16;

    fn f32s(data: &[f32], dims: &[usize]) -> Tensor {
        Tensor::from_slice(data, dims).expect("valid data")
    }

    #[test]
    fn add_same_shape() {
        let sum = f32s(&[1.0, 2.0], &[2])
            .add(&f32s(&[10.0, 20.0], &[2]))
            .expect("add");
        assert_eq!(sum.shape().dims(), &[2]);
        assert_eq!(sum.to_vec::<f32>().expect("f32"), [11.0, 22.0]);
    }

    #[test]
    fn add_broadcasts_a_row_over_a_matrix() {
        let matrix = f32s(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let sum = matrix.add(&f32s(&[10.0, 20.0, 30.0], &[3])).expect("add");
        assert_eq!(sum.shape().dims(), &[2, 3]);
        assert_eq!(
            sum.to_vec::<f32>().expect("f32"),
            [11.0, 22.0, 33.0, 14.0, 25.0, 36.0]
        );
    }

    #[test]
    fn add_broadcasts_both_operands() {
        let column = f32s(&[1.0, 2.0], &[2, 1]);
        let sum = column
            .add(&f32s(&[10.0, 20.0, 30.0], &[1, 3]))
            .expect("add");
        assert_eq!(sum.shape().dims(), &[2, 3]);
        assert_eq!(
            sum.to_vec::<f32>().expect("f32"),
            [11.0, 21.0, 31.0, 12.0, 22.0, 32.0]
        );
    }

    #[test]
    fn add_rejects_non_broadcastable_shapes() {
        let err = f32s(&[0.0; 3], &[3])
            .add(&f32s(&[0.0; 2], &[2]))
            .expect_err("3 vs 2");
        assert!(matches!(
            err,
            Error::NotBroadcastable { ref lhs, ref rhs } if lhs == &[3] && rhs == &[2]
        ));
    }

    #[test]
    fn add_rejects_a_non_f32_lhs() {
        let half = Tensor::from_slice(&[f16::ONE], &[1]).expect("valid data");
        let err = half.add(&f32s(&[1.0], &[1])).expect_err("f16 lhs");
        assert!(matches!(
            err,
            Error::DTypeMismatch {
                expected: DType::F32,
                got: DType::F16
            }
        ));
    }

    #[test]
    fn add_rejects_a_non_f32_rhs() {
        let half = Tensor::from_slice(&[f16::ONE], &[1]).expect("valid data");
        let err = f32s(&[1.0], &[1]).add(&half).expect_err("f16 rhs");
        assert!(matches!(
            err,
            Error::DTypeMismatch {
                expected: DType::F32,
                got: DType::F16
            }
        ));
    }

    #[test]
    fn add_checks_dtypes_before_shapes() {
        let half = Tensor::from_slice(&[f16::ONE; 3], &[3]).expect("valid data");
        let single = f32s(&[0.0; 2], &[2]);
        for err in [
            half.add(&single).expect_err("f16 lhs"),
            single.add(&half).expect_err("f16 rhs"),
        ] {
            assert!(matches!(
                err,
                Error::DTypeMismatch {
                    expected: DType::F32,
                    got: DType::F16
                }
            ));
        }
    }

    #[test]
    fn add_of_scalars() {
        let sum = f32s(&[1.5], &[]).add(&f32s(&[2.0], &[])).expect("add");
        assert_eq!(sum.shape().dims(), &[] as &[usize]);
        assert_eq!(sum.to_vec::<f32>().expect("f32"), [3.5]);
    }
}
