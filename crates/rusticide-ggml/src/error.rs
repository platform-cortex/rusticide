// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

/// Errors raised while building or executing a compute graph.
#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("node {id} not found")]
    NodeNotFound { id: usize },
    #[error("input {input_id} for node {node_id} not found")]
    InputNotFound { node_id: usize, input_id: usize },
    #[error("node {id} is not an input")]
    NotAnInput { id: usize },
    #[error("operator takes {expected} inputs, got {got}")]
    ArityMismatch { expected: usize, got: usize },
    #[error("row index {value} is not a non-negative integer")]
    InvalidIndex { value: f32 },
    #[error(transparent)]
    Tensor(#[from] rusticide_core::Error),
}

/// The crate's result alias.
pub type Result<T> = std::result::Result<T, GraphError>;

#[cfg(test)]
mod tests {
    use super::GraphError;

    #[test]
    fn messages_name_their_fields() {
        assert_eq!(
            GraphError::NodeNotFound { id: 7 }.to_string(),
            "node 7 not found"
        );
        assert_eq!(
            GraphError::InputNotFound {
                node_id: 3,
                input_id: 1
            }
            .to_string(),
            "input 1 for node 3 not found"
        );
        assert_eq!(
            GraphError::NotAnInput { id: 4 }.to_string(),
            "node 4 is not an input"
        );
        assert_eq!(
            GraphError::ArityMismatch {
                expected: 2,
                got: 1
            }
            .to_string(),
            "operator takes 2 inputs, got 1"
        );
        assert_eq!(
            GraphError::InvalidIndex { value: 1.5 }.to_string(),
            "row index 1.5 is not a non-negative integer"
        );
    }
}
