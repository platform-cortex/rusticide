// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

mod add;
mod matmul;
mod reshape;

use crate::{DType, Device, Element, Error, Result, Tensor};

/// Legacy eager CPU operators on [`Tensor`].
///
/// Legacy eager CPU operator (SP0 port of the deleted rusticide-tensor).
/// Bypasses the dispatcher. Replaced by dispatched operators in SP2; do not
/// add new operators here.
pub trait EagerOps: Sized {
    /// Legacy eager CPU operator (SP0 port of the deleted rusticide-tensor).
    /// Bypasses the dispatcher. Replaced by dispatched operators in SP2; do not
    /// add new operators here.
    ///
    /// Reshape to `dims`, row-major. Copies: the result owns new storage
    /// (tier-1 semantics; true views arrive with SP2).
    ///
    /// # Errors
    ///
    /// `Error::ElementCountMismatch` if `dims` holds a different element
    /// count; allocation errors from the copy.
    fn reshape(&self, dims: &[usize]) -> Result<Self>;

    /// Legacy eager CPU operator (SP0 port of the deleted rusticide-tensor).
    /// Bypasses the dispatcher. Replaced by dispatched operators in SP2; do not
    /// add new operators here.
    ///
    /// A contiguous tensor with the same data. Every tensor core can build
    /// is already contiguous with offset 0, so this returns an aliasing
    /// clone; SP2's views must make it copy when that stops holding.
    #[must_use]
    fn contiguous(&self) -> Self;

    /// Legacy eager CPU operator (SP0 port of the deleted rusticide-tensor).
    /// Bypasses the dispatcher. Replaced by dispatched operators in SP2; do not
    /// add new operators here.
    ///
    /// Elementwise sum with `NumPy` broadcasting. Both operands must be `F32`.
    ///
    /// # Errors
    ///
    /// `Error::DTypeMismatch` for a non-`F32` operand (lhs checked first);
    /// `Error::NotBroadcastable` if the shapes do not broadcast; allocation
    /// errors for the output.
    fn add(&self, other: &Self) -> Result<Self>;

    /// Legacy eager CPU operator (SP0 port of the deleted rusticide-tensor).
    /// Bypasses the dispatcher. Replaced by dispatched operators in SP2; do not
    /// add new operators here.
    ///
    /// 2-D matrix product `[m, k] x [k, n] -> [m, n]`. Both operands must be
    /// `F32`; accumulation is in `f32`.
    ///
    /// # Errors
    ///
    /// `Error::DTypeMismatch` for a non-`F32` operand (lhs checked first);
    /// `Error::RankMismatch` for an operand that is not 2-D (lhs first);
    /// `Error::ShapeMismatch` if the inner dimensions differ; allocation errors
    /// for the output.
    fn matmul(&self, other: &Self) -> Result<Self>;
}

impl EagerOps for Tensor {
    fn reshape(&self, dims: &[usize]) -> Result<Self> {
        reshape::reshape(self, dims)
    }

    fn contiguous(&self) -> Self {
        reshape::contiguous(self)
    }

    fn add(&self, other: &Self) -> Result<Self> {
        add::add(self, other)
    }

    fn matmul(&self, other: &Self) -> Result<Self> {
        matmul::matmul(self, other)
    }
}

pub(super) fn require_f32(tensor: &Tensor) -> Result<()> {
    if tensor.dtype() == DType::F32 {
        Ok(())
    } else {
        Err(Error::DTypeMismatch {
            expected: DType::F32,
            got: tensor.dtype(),
        })
    }
}

// `write_ne` is sealed and only callable through an `Element` bound.
fn write_element<T: Element>(value: T, out: &mut [u8]) {
    value.write_ne(out);
}

// `read_ne` is sealed and only callable through an `Element` bound.
fn read_element<T: Element>(bytes: &[u8]) -> T {
    T::read_ne(bytes)
}

/// Element-indexed `f32` view over a tensor's raw bytes, decoded on read.
pub(super) struct F32Reader<'a>(&'a [u8]);

impl F32Reader<'_> {
    pub(super) fn at(&self, flat: usize) -> f32 {
        let size = DType::F32.size_in_bytes();
        read_element(&self.0[flat * size..(flat + 1) * size])
    }
}

/// Run `body` over `f32` readers for both operands, holding one read guard
/// per distinct storage. Two read guards on one `RwLock` from one thread can
/// deadlock when a writer is queued, so an aliasing pair shares one guard.
#[expect(
    clippy::significant_drop_tightening,
    reason = "the guards back the readers handed to `body` and must outlive it"
)]
pub(super) fn with_f32_pair<R>(
    lhs: &Tensor,
    rhs: &Tensor,
    body: impl FnOnce(&F32Reader<'_>, &F32Reader<'_>) -> R,
) -> R {
    if lhs.shares_storage_with(rhs) {
        let guard = lhs.storage().read();
        let reader = F32Reader(&guard);
        return body(&reader, &reader);
    }
    // Lock-order invariant: one guard for shared storage, otherwise the two
    // guards are acquired in ascending `lock_order_key` order, whatever the
    // caller's (lhs, rhs) order, so concurrent `a.op(b)` and `b.op(a)` with
    // queued writers cannot form a wait cycle.
    let lhs_first = lhs.storage().lock_order_key() < rhs.storage().lock_order_key();
    let (first, second) = if lhs_first { (lhs, rhs) } else { (rhs, lhs) };
    let first_guard = first.storage().read();
    let second_guard = second.storage().read();
    let (left, right) = if lhs_first {
        (&first_guard, &second_guard)
    } else {
        (&second_guard, &first_guard)
    };
    body(&F32Reader(left), &F32Reader(right))
}

/// Allocate a CPU `F32` tensor of `dims` fallibly and fill element `flat`
/// with `value_at(flat)` under one write lock.
pub(super) fn filled_f32(dims: &[usize], mut value_at: impl FnMut(usize) -> f32) -> Result<Tensor> {
    let output = Tensor::zeros(dims, DType::F32, Device::cpu())?;
    let mut buffer = output.storage().write();
    for (flat, chunk) in buffer.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        write_element(value_at(flat), chunk.as_mut_slice());
    }
    drop(buffer);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::filled_f32;
    use crate::{DType, Error};

    #[test]
    fn filled_f32_fills_in_flat_order() {
        let t = filled_f32(&[2, 2], |flat| [5.0, 6.0, 7.0, 8.0][flat]).expect("allocation");
        assert_eq!(t.to_vec::<f32>().expect("f32"), [5.0, 6.0, 7.0, 8.0]);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn filled_f32_rejects_an_output_too_large_to_allocate() {
        let err = filled_f32(&[1 << 32, 1 << 32], |_| 0.0).expect_err("overflows usize");
        assert!(matches!(
            err,
            Error::AllocationSizeOverflow {
                numel: usize::MAX,
                dtype: DType::F32
            }
        ));
    }
}
