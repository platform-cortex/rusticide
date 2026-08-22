// SPDX-FileCopyrightText: 2026 Rusticide contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::DType;

/// Errors raised by the core value types.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("dtype mismatch: expected {expected:?}, got {got:?}")]
    DTypeMismatch { expected: DType, got: DType },

    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },

    #[error("shapes {lhs:?} and {rhs:?} are not broadcastable")]
    NotBroadcastable { lhs: Vec<usize>, rhs: Vec<usize> },

    #[error("dimension {dim} is out of range for a {ndim}-D tensor")]
    DimOutOfRange { dim: i64, ndim: usize },

    #[error("allocation of {bytes} bytes failed")]
    AllocationFailed { bytes: usize },
}

/// The crate's result alias.
pub type Result<T> = core::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::Error;
    use crate::DType;

    #[test]
    fn dtype_mismatch_names_both_types() {
        let e = Error::DTypeMismatch {
            expected: DType::F32,
            got: DType::I64,
        };
        let msg = e.to_string();
        assert!(msg.contains("F32"), "message was: {msg}");
        assert!(msg.contains("I64"), "message was: {msg}");
    }

    #[test]
    fn not_broadcastable_contains_both_shapes() {
        let e = Error::NotBroadcastable {
            lhs: vec![2, 3, 4],
            rhs: vec![3, 1],
        };
        let msg = e.to_string();
        assert!(msg.contains("[2, 3, 4]"), "message was: {msg}");
        assert!(msg.contains("[3, 1]"), "message was: {msg}");
    }
}
