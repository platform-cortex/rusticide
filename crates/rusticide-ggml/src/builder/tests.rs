// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use rusticide_core::{DType, Error};

use super::*;
use crate::GraphError;

#[test]
fn invalid_input_id_is_node_not_found() {
    let mut builder = GraphBuilder::new();
    let input = builder.add_input(&[2], DType::F32);
    let err = builder.add(Op::Add, &[input, 999]).unwrap_err();
    assert!(matches!(err, GraphError::NodeNotFound { id: 999 }));
}

#[test]
fn add_input_returns_sequential_ids() {
    let mut builder = GraphBuilder::new();
    assert_eq!(builder.add_input(&[2], DType::F32), 0);
    assert_eq!(builder.add_input(&[2], DType::F32), 1);
    assert_eq!(builder.add(Op::Add, &[0, 1]).unwrap(), 2);
}

#[test]
fn default_equals_new() {
    let mut from_default = GraphBuilder::default();
    let mut from_new = GraphBuilder::new();
    assert_eq!(format!("{from_default:?}"), format!("{from_new:?}"));
    from_default.add_input(&[1], DType::F32);
    from_new.add_input(&[1], DType::F32);
    assert_eq!(format!("{from_default:?}"), format!("{from_new:?}"));
}

#[test]
fn wrong_input_count_is_arity_mismatch() {
    let mut builder = GraphBuilder::new();
    let input = builder.add_input(&[2], DType::F32);
    let err = builder.add(Op::Add, &[input]).unwrap_err();
    assert!(matches!(
        err,
        GraphError::ArityMismatch {
            expected: 2,
            got: 1
        }
    ));
    let err = builder.add(Op::Copy, &[]).unwrap_err();
    assert!(matches!(
        err,
        GraphError::ArityMismatch {
            expected: 1,
            got: 0
        }
    ));
}

#[test]
fn inference_errors_leave_the_builder_unchanged() {
    let mut builder = GraphBuilder::new();
    let lhs = builder.add_input(&[2, 3], DType::F32);
    let rhs = builder.add_input(&[4, 3], DType::F32);
    let err = builder.add(Op::Add, &[lhs, rhs]).unwrap_err();
    assert!(matches!(
        err,
        GraphError::Tensor(Error::NotBroadcastable { .. })
    ));
    assert_eq!(builder.add_input(&[1], DType::F32), 2);
}

#[test]
fn add_reshape_builds_a_reshape_node() {
    let mut builder = GraphBuilder::new();
    let input = builder.add_input(&[2, 3], DType::F32);
    let node = builder.add_reshape(input, &[3, 2]).unwrap();
    let graph = builder.build();
    assert_eq!(
        graph.node(node).unwrap().op(),
        &Op::Reshape { shape: vec![3, 2] }
    );
}

#[test]
fn add_reshape_unknown_input_is_node_not_found() {
    let mut builder = GraphBuilder::new();
    let err = builder.add_reshape(0, &[1]).unwrap_err();
    assert!(matches!(err, GraphError::NodeNotFound { id: 0 }));
}

#[test]
fn add_repeat_builds_a_repeat_node() {
    let mut builder = GraphBuilder::new();
    let input = builder.add_input(&[1, 3], DType::F32);
    let node = builder.add_repeat(input, &[2, 3]).unwrap();
    let graph = builder.build();
    assert_eq!(
        graph.node(node).unwrap().op(),
        &Op::Repeat { shape: vec![2, 3] }
    );
}

#[test]
fn add_repeat_unknown_input_is_node_not_found() {
    let mut builder = GraphBuilder::new();
    let err = builder.add_repeat(3, &[1]).unwrap_err();
    assert!(matches!(err, GraphError::NodeNotFound { id: 3 }));
}
