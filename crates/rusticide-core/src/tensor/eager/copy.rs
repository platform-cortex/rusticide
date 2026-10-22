// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{Result, Shape, Storage, Tensor};

impl Tensor {
    /// A new tensor with its own storage holding a copy of this one's data.
    ///
    /// The copy starts with a fresh version counter.
    ///
    /// # Errors
    ///
    /// Allocation errors from [`Storage::new`].
    pub fn deep_copy(&self) -> Result<Self> {
        self.copy_with_shape(self.shape().clone())
    }

    pub(super) fn copy_with_shape(&self, shape: Shape) -> Result<Self> {
        let storage = Storage::new(self.dtype(), self.device(), self.numel())?;
        storage.write().copy_from_slice(&self.storage().read());
        Ok(Self::from_storage(storage, shape))
    }
}

#[cfg(test)]
mod tests {
    use half::f16;

    use crate::{DType, Device, Tensor};

    #[test]
    fn deep_copy_does_not_alias() {
        let original = Tensor::from_slice(&[1.0f32, 2.0], &[2]).expect("valid data");
        let copy = original.deep_copy().expect("allocation succeeds");
        copy.set(&[0], 9.0f32).expect("in bounds");
        assert_eq!(original.get::<f32>(&[0]).expect("in bounds"), 1.0);
        assert_eq!(copy.to_vec::<f32>().expect("f32"), [9.0, 2.0]);
        assert!(!original.shares_storage_with(&copy));
        assert_eq!(copy.version(), 1);
    }

    #[test]
    fn deep_copy_preserves_dtype_device_and_shape() {
        let source = Tensor::zeros(&[2, 3], DType::F16, Device::gpu(0)).expect("allocation");
        source.set(&[1, 2], f16::from_f32(1.5)).expect("in bounds");
        let copy = source.deep_copy().expect("allocation succeeds");
        assert_eq!(copy.dtype(), DType::F16);
        assert_eq!(copy.device(), Device::gpu(0));
        assert_eq!(copy.shape().dims(), &[2, 3]);
        assert_eq!(
            copy.get::<f16>(&[1, 2]).expect("in bounds"),
            f16::from_f32(1.5)
        );
    }
}
