// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::Op;

/// One operator application in a graph.
#[derive(Debug, Clone)]
pub struct Node {
    pub(crate) id: usize,
    pub(crate) op: Op,
    pub(crate) inputs: Vec<usize>,
}

impl Node {
    #[must_use]
    pub const fn id(&self) -> usize {
        self.id
    }

    #[must_use]
    pub const fn op(&self) -> &Op {
        &self.op
    }

    #[must_use]
    pub fn inputs(&self) -> &[usize] {
        &self.inputs
    }
}
