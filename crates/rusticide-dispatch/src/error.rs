// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

/// Why a dispatcher call failed: no kernel was reached, or the kernel failed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DispatchError {
    /// No operator is registered under this name.
    #[error("unknown operator `{0}`")]
    UnknownOperator(String),
    /// The operator exists but has no kernel for the keyset computed from
    /// the stack, including a stack that carries no tensors at all.
    #[error("operator `{operator}` has no kernel for the dispatch keys of its arguments")]
    NoKernel {
        /// The operator that was called.
        operator: String,
    },
    /// The kernel ran and reported a failure.
    #[error("kernel failed: {0}")]
    KernelFailed(String),
}

impl From<rusticide_core::Error> for DispatchError {
    fn from(error: rusticide_core::Error) -> Self {
        Self::KernelFailed(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::DispatchError;
    use rusticide_core::Error;

    #[test]
    fn errors_name_the_operator() {
        assert_eq!(
            DispatchError::UnknownOperator("add".to_owned()).to_string(),
            "unknown operator `add`"
        );
        let missing = DispatchError::NoKernel {
            operator: "add".to_owned(),
        };
        assert_eq!(
            missing.to_string(),
            "operator `add` has no kernel for the dispatch keys of its arguments"
        );
    }

    #[test]
    fn a_kernel_failure_carries_its_message() {
        assert_eq!(
            DispatchError::KernelFailed("boom".to_owned()).to_string(),
            "kernel failed: boom"
        );
    }

    #[test]
    fn a_core_error_converts_into_a_kernel_failure() {
        let converted = DispatchError::from(Error::AllocationFailed { bytes: 8 });
        assert_eq!(
            converted,
            DispatchError::KernelFailed("allocation of 8 bytes failed".to_owned())
        );
    }
}
