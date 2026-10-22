// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compute graph over `rusticide-core` tensors: a builder that infers and
//! validates node shapes, and a graph that executes its nodes in id order.

mod builder;
mod error;
mod graph;
mod index;
mod infer;
mod node;
mod op;
mod ops;
#[cfg(test)]
mod test_support;

pub use builder::GraphBuilder;
pub use error::{GraphError, Result};
pub use graph::Graph;
pub use node::Node;
pub use op::Op;
