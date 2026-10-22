// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_core::Tensor;

/// Whether every operand is backed by a different allocation.
///
/// `Storage` is an `Arc<RwLock<..>>`, so a call like `a.add_(a)` would
/// take a write and a read lock on one buffer and deadlock. Kernels
/// consult this before locking and degrade to a single write lock when
/// operands coincide. Operand counts are tiny, so the quadratic scan is
/// cheaper than allocating a set.
#[must_use]
pub fn distinct_storages(tensors: &[&Tensor]) -> bool {
    tensors.iter().enumerate().all(|(index, first)| {
        tensors[index + 1..]
            .iter()
            .all(|second| !first.shares_storage_with(second))
    })
}

#[cfg(test)]
mod tests {
    use rusticide_core::{DType, Device, Tensor};

    use super::distinct_storages;

    fn tensor() -> Tensor {
        Tensor::zeros(&[1], DType::F32, Device::cpu()).expect("allocation succeeds")
    }

    #[test]
    fn separate_tensors_have_distinct_storages() {
        let (a, b) = (tensor(), tensor());
        assert!(distinct_storages(&[&a, &b]));
    }

    #[test]
    fn a_tensor_aliased_with_itself_is_detected() {
        let a = tensor();
        let b = a.clone();
        assert!(
            !distinct_storages(&[&a, &b]),
            "a.add_(a) must be seen, or the RwLock deadlocks"
        );
    }

    #[test]
    fn a_single_operand_is_trivially_distinct() {
        let a = tensor();
        assert!(distinct_storages(&[&a]));
    }
}
