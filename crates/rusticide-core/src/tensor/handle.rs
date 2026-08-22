// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::{
    BackendComponent, DType, Device, DeviceKind, DispatchKey, DispatchKeySet, Result, Shape,
    Storage, Strides,
};

use super::impl_::TensorImpl;
use super::version::VersionCounter;

/// A tensor: a shaped, strided view over a storage buffer.
///
/// Cloning a `Tensor` is cheap and aliasing: it clones the `Arc` handle, so
/// clones share the same storage and the same version counter, not copies
/// of them. That is required, not incidental, for the same reason
/// [`Storage`] aliases on clone -- a training loop mutates parameters that
/// a module and an optimizer both hold, and both must observe the update.
#[derive(Debug, Clone)]
pub struct Tensor(Arc<TensorImpl>);

impl Tensor {
    /// Allocate a zero-filled tensor with `dims`, `dtype`, on `device`.
    ///
    /// # Errors
    ///
    /// Returns an error if the backing storage cannot be allocated -- see
    /// [`Storage::new`].
    pub fn zeros(dims: &[usize], dtype: DType, device: Device) -> Result<Self> {
        let shape = Shape::new(dims);
        let strides = Strides::contiguous_for(&shape);
        let storage = Storage::new(dtype, device, shape.numel())?;
        let backend = match device.kind() {
            DeviceKind::Cpu => BackendComponent::Cpu,
            DeviceKind::Gpu => BackendComponent::Gpu,
        };
        let key_set = DispatchKeySet::empty()
            .with_backend(backend)
            .with_functionality(DispatchKey::Dense);
        Ok(Self(Arc::new(TensorImpl {
            storage,
            shape,
            strides,
            storage_offset: 0,
            version: VersionCounter::new(),
            key_set,
        })))
    }

    #[must_use]
    pub fn shape(&self) -> &Shape {
        &self.0.shape
    }

    #[must_use]
    pub fn strides(&self) -> &Strides {
        &self.0.strides
    }

    #[must_use]
    pub fn dtype(&self) -> DType {
        self.0.storage.dtype()
    }

    #[must_use]
    pub fn device(&self) -> Device {
        self.0.storage.device()
    }

    #[must_use]
    pub fn numel(&self) -> usize {
        self.0.shape.numel()
    }

    #[must_use]
    pub fn key_set(&self) -> DispatchKeySet {
        self.0.key_set
    }

    #[must_use]
    pub fn storage(&self) -> &Storage {
        &self.0.storage
    }

    /// The element offset into `storage` at which this tensor's data
    /// begins. Zero for a freshly allocated tensor; a future view/slicing
    /// operation would produce a tensor sharing `storage` with a nonzero
    /// offset instead of copying data.
    #[must_use]
    pub fn storage_offset(&self) -> usize {
        self.0.storage_offset
    }

    #[must_use]
    pub fn version(&self) -> u32 {
        self.0.version.get()
    }

    /// Records an in-place mutation. See [`VersionCounter`].
    pub fn bump_version(&self) {
        self.0.version.bump();
    }

    #[must_use]
    pub fn shares_storage_with(&self, other: &Self) -> bool {
        self.0.storage.same_buffer_as(&other.0.storage)
    }
}

#[cfg(test)]
mod tests {
    use super::Tensor;
    use crate::{BackendComponent, DType, Device, DispatchKey};

    #[test]
    fn zeros_has_the_requested_shape_and_dtype() {
        let t = Tensor::zeros(&[2, 3], DType::F32, Device::cpu()).expect("allocation succeeds");
        assert_eq!(t.shape().dims(), &[2, 3]);
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.strides().as_slice(), &[3, 1]);
    }

    #[test]
    fn a_cpu_dense_tensor_carries_the_matching_keys() {
        let t = Tensor::zeros(&[1], DType::F32, Device::cpu()).expect("allocation succeeds");
        let ks = t.key_set();
        assert!(ks.has_backend(BackendComponent::Cpu));
        assert_eq!(ks.highest_functionality(), Some(DispatchKey::Dense));
    }

    #[test]
    fn cloning_a_tensor_shares_storage() {
        let a = Tensor::zeros(&[4], DType::U8, Device::cpu()).expect("allocation succeeds");
        let b = a.clone();
        assert!(a.shares_storage_with(&b));
    }

    #[test]
    fn version_starts_at_zero_and_bumps_on_mutation() {
        let t = Tensor::zeros(&[1], DType::F32, Device::cpu()).expect("allocation succeeds");
        assert_eq!(t.version(), 0);
        t.bump_version();
        assert_eq!(t.version(), 1);
    }

    #[test]
    fn shared_tensors_share_one_version_counter() {
        let a = Tensor::zeros(&[1], DType::F32, Device::cpu()).expect("allocation succeeds");
        let b = a.clone();
        a.bump_version();
        assert_eq!(b.version(), 1, "autograd relies on this");
    }

    /// The dispatcher (Task 16) selects kernels by reading the backend bit
    /// off a tensor's own key set. If a tensor built for one device carried
    /// the wrong backend bit -- or the same bits as a tensor on another
    /// device -- dispatch would silently route to the wrong device's
    /// kernel instead of failing loudly.
    #[test]
    fn gpu_and_cpu_tensors_carry_distinct_backend_keysets() {
        let cpu = Tensor::zeros(&[1], DType::F32, Device::cpu()).expect("allocation succeeds");
        let gpu = Tensor::zeros(&[1], DType::F32, Device::gpu(0)).expect("allocation succeeds");

        assert!(cpu.key_set().has_backend(BackendComponent::Cpu));
        assert!(!cpu.key_set().has_backend(BackendComponent::Gpu));

        assert!(gpu.key_set().has_backend(BackendComponent::Gpu));
        assert!(!gpu.key_set().has_backend(BackendComponent::Cpu));

        assert_ne!(cpu.key_set(), gpu.key_set());
    }
}
