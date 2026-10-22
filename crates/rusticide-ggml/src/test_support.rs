// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_core::{DType, Tensor};

use crate::{Graph, GraphBuilder, Op, Result};

pub(crate) fn f32_tensor(data: &[f32], dims: &[usize]) -> Tensor {
    Tensor::from_slice(data, dims).unwrap()
}

pub(crate) fn assert_values(tensor: &Tensor, dims: &[usize], expected: &[f32]) {
    assert_eq!(tensor.shape().dims(), dims);
    assert_eq!(tensor.to_vec::<f32>().unwrap(), expected);
}

/// Builds a one-op graph whose inputs are declared as `declared` but fed
/// `tensors`, runs it, and returns the op's result.
pub(crate) fn run_declared(
    op: Op,
    declared: &[(&[usize], DType)],
    tensors: &[Tensor],
) -> Result<Tensor> {
    let mut builder = GraphBuilder::new();
    let ids: Vec<usize> = declared
        .iter()
        .map(|&(dims, dtype)| builder.add_input(dims, dtype))
        .collect();
    let node = builder.add(op, &ids)?;
    let mut graph: Graph = builder.build();
    for (&id, tensor) in ids.iter().zip(tensors) {
        graph.set_input(id, tensor.clone())?;
    }
    graph.forward()?;
    Ok(graph.tensor(node).unwrap().clone())
}

pub(crate) fn run(op: Op, tensors: &[Tensor]) -> Result<Tensor> {
    let declared: Vec<(&[usize], DType)> = tensors
        .iter()
        .map(|tensor| (tensor.shape().dims(), tensor.dtype()))
        .collect();
    run_declared(op, &declared, tensors)
}

pub(crate) fn tensor_error(err: crate::GraphError) -> rusticide_core::Error {
    let crate::GraphError::Tensor(inner) = err else {
        panic!("expected a tensor error, got {err:?}");
    };
    inner
}
