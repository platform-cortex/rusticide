// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{Error, Result};

/// An empty `Vec<T>` with room for exactly `capacity` elements, reserved
/// fallibly.
///
/// Workspace-internal; removed once SP2 decodes directly into Tensor storage.
///
/// # Errors
///
/// `Error::AllocationFailed` if the allocator refuses the request or
/// `capacity * size_of::<T>()` exceeds the address space. `bytes` saturates
/// at `usize::MAX` in the latter case.
#[doc(hidden)]
pub fn try_vec_with_capacity<T>(capacity: usize) -> Result<Vec<T>> {
    let mut vec = Vec::new();
    vec.try_reserve_exact(capacity)
        .map_err(|_| Error::AllocationFailed {
            bytes: capacity.saturating_mul(size_of::<T>()),
        })?;
    Ok(vec)
}

#[cfg(test)]
mod tests {
    use super::try_vec_with_capacity;
    use crate::Error;

    #[test]
    fn reserves_exactly_the_requested_capacity_and_starts_empty() {
        let v = try_vec_with_capacity::<f32>(7).expect("small reservation succeeds");
        assert_eq!(v.len(), 0);
        assert!(v.capacity() >= 7);
    }

    #[test]
    fn an_unsatisfiable_reservation_reports_the_byte_count_not_a_panic() {
        let err = try_vec_with_capacity::<u32>(usize::MAX).expect_err("exceeds the address space");
        assert!(
            matches!(err, Error::AllocationFailed { bytes: usize::MAX }),
            "expected AllocationFailed with a saturated byte count, got: {err:?}"
        );
    }

    #[test]
    fn the_reported_bytes_are_count_times_element_size() {
        let count = usize::MAX / 2;
        let err = try_vec_with_capacity::<u16>(count).expect_err("exceeds isize::MAX bytes");
        assert!(
            matches!(err, Error::AllocationFailed { bytes } if bytes == count * 2),
            "got: {err:?}"
        );
    }
}
