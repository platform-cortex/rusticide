// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use rusticide_alloc::Buffer;

use crate::{DType, Device, Error, Result};

/// The buffer backing one or more tensors.
///
/// Cloning a `Storage` aliases the same buffer. That is required, not
/// incidental: a training loop mutates parameters that the module and
/// the optimizer both hold, and both must observe the update. See
/// "Aliasing and mutation" in the design document.
#[derive(Debug, Clone)]
pub struct Storage {
    buffer: Arc<RwLock<Buffer>>,
    dtype: DType,
    device: Device,
    numel: usize,
}

impl Storage {
    /// Allocate zeroed storage for `numel` elements of `dtype` on `device`.
    ///
    /// # Errors
    ///
    /// Returns `Error::AllocationSizeOverflow` if `numel * dtype.size_in_bytes()`
    /// overflows `usize` -- no allocation is attempted in that case, since
    /// there is no representable byte count to request. Returns
    /// `Error::AllocationFailed` if a representable request is refused by
    /// the allocator.
    pub fn new(dtype: DType, device: Device, numel: usize) -> Result<Self> {
        let bytes = numel
            .checked_mul(dtype.size_in_bytes())
            .ok_or(Error::AllocationSizeOverflow { numel, dtype })?;
        let buffer = Buffer::alloc_zeroed(bytes).ok_or(Error::AllocationFailed { bytes })?;
        Ok(Self {
            buffer: Arc::new(RwLock::new(buffer)),
            dtype,
            device,
            numel,
        })
    }

    /// A shared view of the bytes.
    ///
    /// # Panics
    ///
    /// Never panics. If the lock is poisoned -- which can only happen if
    /// another thread panicked while holding it -- the poisoned guard is
    /// recovered via `into_inner` rather than propagated. A half-written
    /// tensor has no meaningful recovery, and propagating a `Result` here
    /// would force every kernel call site to plumb an error it cannot act
    /// on differently.
    pub fn read(&self) -> RwLockReadGuard<'_, Buffer> {
        match self.buffer.read() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    /// An exclusive view of the bytes.
    ///
    /// See [`Storage::read`] for why lock poisoning is recovered rather
    /// than propagated.
    pub fn write(&self) -> RwLockWriteGuard<'_, Buffer> {
        match self.buffer.write() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    #[must_use]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    #[must_use]
    pub const fn device(&self) -> Device {
        self.device
    }

    #[must_use]
    pub const fn numel(&self) -> usize {
        self.numel
    }

    /// Whether two storages are backed by the same allocation.
    ///
    /// The dispatcher uses this to detect self-aliasing calls such as
    /// `a.add_(a)`, which would otherwise deadlock by taking a read lock
    /// and a write lock on the same buffer.
    #[must_use]
    pub fn same_buffer_as(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.buffer, &other.buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::Storage;
    use crate::{DType, Device, Error};

    #[test]
    fn new_storage_is_sized_from_dtype_and_count() {
        let s = Storage::new(DType::F32, Device::cpu(), 10).expect("allocation succeeds");
        assert_eq!(s.numel(), 10);
        assert_eq!(s.read().len(), 40);
    }

    #[test]
    fn clones_share_one_buffer() {
        let a = Storage::new(DType::U8, Device::cpu(), 4).expect("allocation succeeds");
        let b = a.clone();
        b.write()[0] = 9;
        assert_eq!(a.read()[0], 9, "the clone must alias, not copy");
        assert!(a.same_buffer_as(&b));
    }

    #[test]
    fn distinct_allocations_do_not_alias() {
        let a = Storage::new(DType::U8, Device::cpu(), 4).expect("allocation succeeds");
        let b = Storage::new(DType::U8, Device::cpu(), 4).expect("allocation succeeds");
        assert!(!a.same_buffer_as(&b));
    }

    #[test]
    fn write_visibility_agrees_with_same_buffer_as() {
        let a = Storage::new(DType::U8, Device::cpu(), 4).expect("allocation succeeds");
        let b = a.clone();
        let c = Storage::new(DType::U8, Device::cpu(), 4).expect("allocation succeeds");

        b.write()[0] = 42;

        assert!(a.same_buffer_as(&b));
        assert_eq!(
            a.read()[0],
            42,
            "same_buffer_as says these alias, so a write through b must be visible through a"
        );

        assert!(!a.same_buffer_as(&c));
        assert_ne!(
            a.read()[0],
            c.read()[0],
            "same_buffer_as says these do not alias, so a's write must not reach c"
        );
    }

    #[test]
    fn overflowing_size_reports_overflow_not_a_bogus_allocation() {
        // usize::MAX is the reproduction case from the review finding: with
        // F64 (8 bytes/element), numel * size_in_bytes overflows usize, so
        // no allocation is ever attempted.
        let err = Storage::new(DType::F64, Device::cpu(), usize::MAX)
            .expect_err("numel * size_in_bytes overflows usize");

        assert!(
            matches!(err, Error::AllocationSizeOverflow { .. }),
            "expected AllocationSizeOverflow, got: {err:?}"
        );

        let msg = err.to_string();
        // The old bug's exact wrong phrasing -- "allocation of N bytes
        // failed" -- claimed a real allocation request was refused. Neither
        // fragment may appear: no allocation was attempted, so nothing
        // "failed" to allocate.
        assert!(
            !msg.contains("allocation of"),
            "message must not claim an allocation was attempted: {msg}"
        );
        assert!(
            !msg.contains("bytes failed"),
            "message must not use the old bytes-failed phrasing: {msg}"
        );
        // The new message is honest: it reports numel/dtype, the actual
        // inputs, not a derived byte count. Confirmed structurally rather
        // than by matching the literal usize::MAX digit string, which
        // coincides with numel itself for this particular reproduction
        // input and so cannot distinguish "reports numel honestly" from
        // "reports a bogus byte count" on its own.
        assert!(msg.contains("elements"), "message was: {msg}");
        assert!(msg.contains("F64"), "message was: {msg}");

        // A numel that overflows without being usize::MAX confirms the
        // reported number really does track the request, not a fixed
        // sentinel like the old bug's `bytes: usize::MAX`.
        let distinct_numel = usize::MAX / 2;
        let err = Storage::new(DType::F64, Device::cpu(), distinct_numel)
            .expect_err("numel * size_in_bytes still overflows usize");
        let msg = err.to_string();
        assert!(
            msg.contains(&distinct_numel.to_string()),
            "message must report the actual numel that was requested: {msg}"
        );
        assert!(
            !msg.contains("18446744073709551615"),
            "message must not report an unrelated sentinel byte count: {msg}"
        );
    }
}
