// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::Storage;

impl Storage {
    /// Runs `body` over the bytes of `self` and `other`, holding read guards
    /// on both.
    ///
    /// Two read guards on one `RwLock` from one thread can deadlock when a
    /// writer is queued, so storages that alias share a single guard. For
    /// distinct storages the guards are acquired in ascending
    /// `lock_order_key` order whatever the argument order, so concurrent
    /// `a.with_read_pair(b)` and `b.with_read_pair(a)` cannot form a wait
    /// cycle. Kernels needing two operands at once must use this method.
    pub fn with_read_pair<R>(&self, other: &Self, body: impl FnOnce(&[u8], &[u8]) -> R) -> R {
        if self.same_buffer_as(other) {
            let guard = self.read();
            return body(&guard, &guard);
        }
        let self_first = self.lock_order_key() < other.lock_order_key();
        let (first, second) = if self_first {
            (self, other)
        } else {
            (other, self)
        };
        let first_guard = first.read();
        let second_guard = second.read();
        if self_first {
            body(&first_guard, &second_guard)
        } else {
            body(&second_guard, &first_guard)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use super::Storage;
    use crate::{DType, Device};

    fn storage_with(byte: u8) -> Storage {
        let storage = Storage::new(DType::U8, Device::cpu(), 4).expect("allocation succeeds");
        storage.write().fill(byte);
        storage
    }

    fn ordered_by_address() -> (Storage, Storage) {
        let (a, b) = (storage_with(1), storage_with(2));
        if a.lock_order_key() < b.lock_order_key() {
            (a, b)
        } else {
            (b, a)
        }
    }

    #[test]
    fn the_body_sees_self_then_other_whatever_the_addresses() {
        let (low, high) = ordered_by_address();
        let from_low = low.with_read_pair(&high, |a, b| (a[0], b[0]));
        let from_high = high.with_read_pair(&low, |a, b| (a[0], b[0]));
        assert_eq!(from_low, (low.read()[0], high.read()[0]));
        assert_eq!(from_high, (high.read()[0], low.read()[0]));
    }

    #[test]
    fn an_aliasing_pair_shares_one_guard() {
        let a = storage_with(7);
        let alias = a.clone();
        let (same, bytes) = a.with_read_pair(&alias, |left, right| {
            (left.as_ptr() == right.as_ptr(), left[0])
        });
        assert!(same);
        assert_eq!(bytes, 7);
    }

    #[test]
    fn the_lower_address_is_locked_first_even_when_passed_second() {
        let (low, high) = ordered_by_address();
        let low = Arc::new(low);
        let blocker = high.write();
        let worker = {
            let (low, high) = (Arc::clone(&low), high.clone());
            std::thread::spawn(move || high.with_read_pair(&low, |_, _| ()))
        };
        let deadline = Instant::now() + Duration::from_secs(5);
        while low.buffer.try_write().is_ok() && Instant::now() < deadline {
            std::thread::yield_now();
        }
        let low_is_read_locked = low.buffer.try_write().is_err();
        drop(blocker);
        worker
            .join()
            .expect("worker finishes once the writer is gone");
        assert!(
            low_is_read_locked,
            "the worker must hold `low` while waiting on `high`"
        );
    }
}
