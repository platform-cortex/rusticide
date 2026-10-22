// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

/// Why a dispatcher call could not reach a kernel.
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
}

#[cfg(test)]
mod tests {
    use super::DispatchError;

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
}
