// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_core::DType;

use crate::infer::infer_shape_dtype;
use crate::{Graph, GraphError, Node, Op, Result};

/// Declares a graph's inputs and operators, validating each as it is added.
#[derive(Debug, Clone)]
pub struct GraphBuilder {
    nodes: Vec<Node>,
    shapes: Vec<Vec<usize>>,
    dtypes: Vec<DType>,
}

impl GraphBuilder {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            nodes: Vec::new(),
            shapes: Vec::new(),
            dtypes: Vec::new(),
        }
    }

    /// Add a placeholder whose tensor is supplied through
    /// [`Graph::set_input`].
    pub fn add_input(&mut self, shape: &[usize], dtype: DType) -> usize {
        self.push(Op::Copy, Vec::new(), shape.to_vec(), dtype)
    }

    /// Add an operator node over earlier nodes.
    ///
    /// # Errors
    ///
    /// `GraphError::NodeNotFound` if an input id does not name an earlier
    /// node; `GraphError::ArityMismatch` for a wrong input count; the
    /// tensor error execution would raise for incompatible shapes or dtypes.
    pub fn add(&mut self, op: Op, inputs: &[usize]) -> Result<usize> {
        if let Some(&id) = inputs.iter().find(|&&id| id >= self.nodes.len()) {
            return Err(GraphError::NodeNotFound { id });
        }
        if inputs.len() != op.arity() {
            return Err(GraphError::ArityMismatch {
                expected: op.arity(),
                got: inputs.len(),
            });
        }
        let described: Vec<(&[usize], DType)> = inputs
            .iter()
            .map(|&id| (self.shapes[id].as_slice(), self.dtypes[id]))
            .collect();
        let (shape, dtype) = infer_shape_dtype(&op, &described)?;
        Ok(self.push(op, inputs.to_vec(), shape, dtype))
    }

    /// Add a reshape of `input` to `shape`.
    ///
    /// # Errors
    ///
    /// As [`GraphBuilder::add`].
    pub fn add_reshape(&mut self, input: usize, shape: &[usize]) -> Result<usize> {
        self.add(
            Op::Reshape {
                shape: shape.to_vec(),
            },
            &[input],
        )
    }

    /// Add a broadcast of `input` up to `shape`.
    ///
    /// # Errors
    ///
    /// As [`GraphBuilder::add`].
    pub fn add_repeat(&mut self, input: usize, shape: &[usize]) -> Result<usize> {
        self.add(
            Op::Repeat {
                shape: shape.to_vec(),
            },
            &[input],
        )
    }

    #[must_use]
    pub fn build(self) -> Graph {
        let tensors = vec![None; self.nodes.len()];
        Graph {
            nodes: self.nodes,
            tensors,
        }
    }

    fn push(&mut self, op: Op, inputs: Vec<usize>, shape: Vec<usize>, dtype: DType) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node { id, op, inputs });
        self.shapes.push(shape);
        self.dtypes.push(dtype);
        id
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
