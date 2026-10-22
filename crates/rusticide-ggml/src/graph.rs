// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_core::Tensor;

use crate::ops::execute_op;
use crate::{GraphError, Node, Result};

/// A built compute graph and the tensors computed so far.
#[derive(Debug, Clone)]
pub struct Graph {
    pub(crate) nodes: Vec<Node>,
    pub(crate) tensors: Vec<Option<Tensor>>,
}

impl Graph {
    #[must_use]
    pub fn node(&self, id: usize) -> Option<&Node> {
        self.nodes.get(id)
    }

    #[must_use]
    pub fn tensor(&self, id: usize) -> Option<&Tensor> {
        self.tensors.get(id).and_then(Option::as_ref)
    }

    /// Supply the tensor of input node `node_id`, replacing any earlier one.
    ///
    /// # Errors
    ///
    /// `GraphError::NodeNotFound` if `node_id` names no node;
    /// `GraphError::NotAnInput` if it names an operator node, whose tensor
    /// only `forward` computes.
    pub fn set_input(&mut self, node_id: usize, tensor: Tensor) -> Result<()> {
        let node = self
            .nodes
            .get(node_id)
            .ok_or(GraphError::NodeNotFound { id: node_id })?;
        if !node.inputs.is_empty() {
            return Err(GraphError::NotAnInput { id: node_id });
        }
        self.tensors[node_id] = Some(tensor);
        Ok(())
    }

    /// Recompute every operator node, in id order, from the inputs set now.
    ///
    /// Inputs always precede their node, so id order is a topological
    /// order. Results of an earlier call are replaced, never reused. Input
    /// nodes keep the tensors set on them.
    ///
    /// # Errors
    ///
    /// `GraphError::InputNotFound` if a node's input has no tensor; the
    /// error an operator raises on the tensors it is given.
    pub fn forward(&mut self) -> Result<()> {
        for (node_id, node) in self.nodes.iter().enumerate() {
            if node.inputs.is_empty() {
                continue;
            }
            let inputs = node
                .inputs
                .iter()
                .map(|&input_id| {
                    self.tensors[input_id]
                        .as_ref()
                        .ok_or(GraphError::InputNotFound { node_id, input_id })
                })
                .collect::<Result<Vec<&Tensor>>>()?;
            let result = execute_op(&node.op, &inputs)?;
            self.tensors[node_id] = Some(result);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
