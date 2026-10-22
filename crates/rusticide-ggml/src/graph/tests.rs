// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_core::DType;

use crate::test_support::{assert_values, f32_tensor};
use crate::{GraphBuilder, GraphError, Op};

#[test]
fn set_input_unknown_node() {
    let mut graph = GraphBuilder::new().build();
    let err = graph.set_input(5, f32_tensor(&[1.0], &[1])).unwrap_err();
    assert!(matches!(err, GraphError::NodeNotFound { id: 5 }));
}

#[test]
fn input_not_set_is_input_not_found() {
    let mut builder = GraphBuilder::new();
    let input = builder.add_input(&[3], DType::F32);
    let copy = builder.add(Op::Copy, &[input]).unwrap();
    let mut graph = builder.build();
    let err = graph.forward().unwrap_err();
    assert!(matches!(
        err,
        GraphError::InputNotFound {
            node_id,
            input_id
        } if node_id == copy && input_id == input
    ));
}

#[test]
fn empty_graph_forward_is_ok() {
    let mut graph = GraphBuilder::new().build();
    graph.forward().unwrap();
}

#[test]
fn forward_skips_already_computed_nodes() {
    let mut builder = GraphBuilder::new();
    let input = builder.add_input(&[2], DType::F32);
    let copy = builder.add(Op::Copy, &[input]).unwrap();
    let mut graph = builder.build();
    graph
        .set_input(copy, f32_tensor(&[7.0, 8.0], &[2]))
        .unwrap();
    graph.forward().unwrap();
    assert_values(graph.tensor(copy).unwrap(), &[2], &[7.0, 8.0]);
}

#[test]
fn forward_executes_in_id_order() {
    let mut builder = GraphBuilder::new();
    let input = builder.add_input(&[2, 3], DType::F32);
    let reshaped = builder.add_reshape(input, &[3, 2]).unwrap();
    let copied = builder.add(Op::Copy, &[reshaped]).unwrap();
    let mut graph = builder.build();
    graph
        .set_input(input, f32_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();
    graph.forward().unwrap();
    assert_values(
        graph.tensor(copied).unwrap(),
        &[3, 2],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
}

#[test]
fn node_accessors() {
    let mut builder = GraphBuilder::new();
    let input = builder.add_input(&[2], DType::F32);
    let copy = builder.add(Op::Copy, &[input]).unwrap();
    let mut graph = builder.build();

    let node = graph.node(copy).unwrap();
    assert_eq!(node.id(), copy);
    assert_eq!(node.op(), &Op::Copy);
    assert_eq!(node.inputs(), &[input]);
    assert!(graph.node(99).is_none());

    assert!(graph.tensor(input).is_none());
    graph
        .set_input(input, f32_tensor(&[1.0, 2.0], &[2]))
        .unwrap();
    assert!(graph.tensor(input).is_some());
    assert!(graph.tensor(99).is_none());
}
