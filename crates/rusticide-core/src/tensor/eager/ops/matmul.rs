// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{filled_f32, require_f32, with_f32_pair};
use crate::{Error, Result, Tensor};

/// Legacy eager CPU operator (SP0 port of the deleted rusticide-tensor).
/// Bypasses the dispatcher. Replaced by dispatched operators in SP2; do not
/// add new operators here.
pub(super) fn matmul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    require_f32(lhs)?;
    require_f32(rhs)?;
    require_rank_two(lhs)?;
    require_rank_two(rhs)?;
    let (left_dims, right_dims) = (lhs.shape().dims(), rhs.shape().dims());
    if left_dims[1] != right_dims[0] {
        return Err(Error::ShapeMismatch {
            expected: left_dims.to_vec(),
            got: right_dims.to_vec(),
        });
    }
    let (inner, columns) = (left_dims[1], right_dims[1]);
    with_f32_pair(lhs, rhs, |left, right| {
        filled_f32(&[left_dims[0], columns], |flat| {
            let (row, column) = (flat / columns, flat % columns);
            (0..inner)
                .map(|k| left.at(row * inner + k) * right.at(k * columns + column))
                .fold(0.0_f32, |sum, term| sum + term)
        })
    })
}

fn require_rank_two(tensor: &Tensor) -> Result<()> {
    if tensor.ndim() == 2 {
        Ok(())
    } else {
        Err(Error::RankMismatch {
            expected: 2,
            got: tensor.ndim(),
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{DType, EagerOps, Error, Tensor};
    use half::f16;

    fn f32s(data: &[f32], dims: &[usize]) -> Tensor {
        Tensor::from_slice(data, dims).expect("valid data")
    }

    #[test]
    fn matmul_2x3_by_3x2() {
        let lhs = f32s(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let rhs = f32s(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);
        let product = lhs.matmul(&rhs).expect("matmul");
        assert_eq!(product.shape().dims(), &[2, 2]);
        assert_eq!(
            product.to_vec::<f32>().expect("f32"),
            [58.0, 64.0, 139.0, 154.0]
        );
    }

    #[test]
    fn matmul_of_a_tensor_with_itself() {
        let a = f32s(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let product = a.matmul(&a).expect("matmul");
        assert_eq!(product.shape().dims(), &[2, 2]);
        assert_eq!(
            product.to_vec::<f32>().expect("f32"),
            [7.0, 10.0, 15.0, 22.0]
        );
        assert_eq!(a.to_vec::<f32>().expect("f32"), [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn matmul_is_not_commutative_in_either_operand_order() {
        let a = f32s(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = f32s(&[0.0, 1.0, 1.0, 0.0], &[2, 2]);
        assert_eq!(
            a.matmul(&b).expect("matmul").to_vec::<f32>().expect("f32"),
            [2.0, 1.0, 4.0, 3.0]
        );
        assert_eq!(
            b.matmul(&a).expect("matmul").to_vec::<f32>().expect("f32"),
            [3.0, 4.0, 1.0, 2.0]
        );
    }

    #[test]
    fn matmul_rejects_a_1d_lhs() {
        let err = f32s(&[1.0, 2.0], &[2])
            .matmul(&f32s(&[1.0, 2.0], &[2, 1]))
            .expect_err("1-D");
        assert!(matches!(
            err,
            Error::RankMismatch {
                expected: 2,
                got: 1
            }
        ));
    }

    #[test]
    fn matmul_rejects_a_3d_rhs() {
        let err = f32s(&[1.0, 2.0], &[1, 2])
            .matmul(&f32s(&[0.0; 4], &[2, 1, 2]))
            .expect_err("3-D");
        assert!(matches!(
            err,
            Error::RankMismatch {
                expected: 2,
                got: 3
            }
        ));
    }

    #[test]
    fn matmul_rejects_mismatched_inner_dims() {
        let err = f32s(&[0.0; 6], &[2, 3])
            .matmul(&f32s(&[0.0; 8], &[4, 2]))
            .expect_err("3 != 4");
        assert!(matches!(
            err,
            Error::ShapeMismatch { ref expected, ref got } if expected == &[2, 3] && got == &[4, 2]
        ));
    }

    #[test]
    fn matmul_rejects_non_f32() {
        let half = Tensor::from_slice(&[f16::ONE], &[1, 1]).expect("valid data");
        let single = f32s(&[1.0], &[1, 1]);
        for err in [
            half.matmul(&single).expect_err("f16 lhs"),
            single.matmul(&half).expect_err("f16 rhs"),
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
    fn matmul_checks_dtypes_before_shapes() {
        let half_vector = Tensor::from_slice(&[f16::ONE; 2], &[2]).expect("valid data");
        let half_inner = Tensor::from_slice(&[f16::ONE; 8], &[4, 2]).expect("valid data");
        let single = f32s(&[0.0; 6], &[2, 3]);
        for err in [
            half_vector.matmul(&single).expect_err("f16 1-D lhs"),
            single.matmul(&half_inner).expect_err("f16 rhs, bad inner"),
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
    fn matmul_of_an_empty_inner_dim_is_zeros() {
        let product = f32s(&[], &[2, 0])
            .matmul(&f32s(&[], &[0, 3]))
            .expect("matmul");
        assert!(
            product
                .to_vec::<f32>()
                .expect("f32")
                .iter()
                .all(|value| value.to_bits() == 0.0_f32.to_bits())
        );
    }

    #[test]
    fn matmul_with_distinct_m_k_n() {
        let lhs = f32s(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let rhs = f32s(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[3, 4],
        );
        let product = lhs.matmul(&rhs).expect("matmul");
        assert_eq!(product.shape().dims(), &[2, 4]);
        assert_eq!(
            product.to_vec::<f32>().expect("f32"),
            [38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0]
        );
    }
}
