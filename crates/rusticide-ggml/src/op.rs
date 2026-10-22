// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

/// A graph operator. The math operators arrive with a later task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    Copy,
    Reshape { shape: Vec<usize> },
    Permute,
    GetRows,
    Add,
    MatMul,
    Cont,
    Repeat { shape: Vec<usize> },
}

impl Op {
    pub(crate) const fn arity(&self) -> usize {
        match self {
            Self::Copy
            | Self::Reshape { .. }
            | Self::Permute
            | Self::Cont
            | Self::Repeat { .. } => 1,
            Self::GetRows | Self::Add | Self::MatMul => 2,
        }
    }
}
