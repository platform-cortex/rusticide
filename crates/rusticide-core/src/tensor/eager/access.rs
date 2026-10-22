// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Legacy eager / SP0 port: building tensors from slices and reading or
//! writing single elements.
//!
//! Data access is legitimate core API; the operators live elsewhere. Every
//! tensor core can build today is contiguous with `storage_offset == 0`, and
//! `to_vec` reads storage linearly relying on that. SP2's views must revisit
//! it.

use crate::{Device, Element, Error, Result, Shape, Storage, Tensor};

impl Tensor {
    /// Build a contiguous CPU tensor of `dims` from `data`, row-major.
    /// `dims = &[]` is a scalar.
    ///
    /// # Errors
    ///
    /// `Error::ElementCountMismatch` if `data.len()` is not the element count
    /// of `dims`; allocation errors from [`Storage::new`].
    pub fn from_slice<T: Element>(data: &[T], dims: &[usize]) -> Result<Self> {
        let shape = Shape::new(dims);
        if shape.numel() != data.len() {
            return Err(Error::ElementCountMismatch {
                expected: shape.numel(),
                got: data.len(),
            });
        }
        let storage = Storage::new(T::DTYPE, Device::cpu(), data.len())?;
        let mut buffer = storage.write();
        for (value, chunk) in data
            .iter()
            .zip(buffer.chunks_exact_mut(T::DTYPE.size_in_bytes()))
        {
            value.write_ne(chunk);
        }
        drop(buffer);
        Ok(Self::from_storage(storage, shape))
    }

    /// Read the element at `index`.
    ///
    /// Relies on the contiguity invariant in the module docs (strides from
    /// a contiguous layout, `storage_offset == 0`).
    ///
    /// # Errors
    ///
    /// `Error::DTypeMismatch` if `T` is not this tensor's dtype;
    /// `Error::IndexOutOfBounds` if `index` has the wrong rank or a
    /// coordinate past its extent.
    pub fn get<T: Element>(&self, index: &[usize]) -> Result<T> {
        self.check_dtype::<T>()?;
        let size = T::DTYPE.size_in_bytes();
        let start = self.element_offset(index)? * size;
        Ok(T::read_ne(&self.storage().read()[start..start + size]))
    }

    /// Write `value` at `index` in place, then bump the version counter.
    ///
    /// Relies on the contiguity invariant in the module docs (strides from
    /// a contiguous layout, `storage_offset == 0`).
    ///
    /// # Errors
    ///
    /// Same as [`Tensor::get`]; a failed call is not a mutation and leaves
    /// the version untouched.
    pub fn set<T: Element>(&self, index: &[usize], value: T) -> Result<()> {
        self.check_dtype::<T>()?;
        let size = T::DTYPE.size_in_bytes();
        let start = self.element_offset(index)? * size;
        value.write_ne(&mut self.storage().write()[start..start + size]);
        self.bump_version();
        Ok(())
    }

    /// Copy every element out in row-major order.
    ///
    /// Relies on the contiguity invariant in the module docs.
    ///
    /// # Errors
    ///
    /// `Error::DTypeMismatch` if `T` is not this tensor's dtype.
    pub fn to_vec<T: Element>(&self) -> Result<Vec<T>> {
        self.check_dtype::<T>()?;
        let buffer = self.storage().read();
        Ok(buffer
            .chunks_exact(T::DTYPE.size_in_bytes())
            .map(T::read_ne)
            .collect())
    }

    /// Number of dimensions; 0 for a scalar.
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.shape().ndim()
    }

    fn check_dtype<T: Element>(&self) -> Result<()> {
        if T::DTYPE == self.dtype() {
            Ok(())
        } else {
            Err(Error::DTypeMismatch {
                expected: T::DTYPE,
                got: self.dtype(),
            })
        }
    }

    fn element_offset(&self, index: &[usize]) -> Result<usize> {
        let dims = self.shape().dims();
        if index.len() != dims.len() || index.iter().zip(dims).any(|(at, extent)| at >= extent) {
            return Err(Error::IndexOutOfBounds {
                index: index.to_vec(),
                shape: dims.to_vec(),
            });
        }
        Ok(index
            .iter()
            .zip(self.strides().as_slice())
            .map(|(at, stride)| at * stride)
            .sum())
    }
}

#[cfg(test)]
mod tests;
