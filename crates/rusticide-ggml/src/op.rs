// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

/// A graph operator.
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    Copy,
    Reshape {
        shape: Vec<usize>,
    },
    Permute,
    GetRows,
    Add,
    MatMul,
    Cont,
    Repeat {
        shape: Vec<usize>,
    },
    Mul,
    Scale(f32),
    SoftMax,
    /// Normalises each row by its root mean square, `eps` added under the root.
    RMSNorm {
        eps: f32,
    },
    /// Rotates the first `dim_pairs` adjacent column pairs of a 2-D input by
    /// an angle of `row * theta^(-2 pair / columns)`.
    RoPE {
        theta: f32,
        dim_pairs: usize,
    },
    Silu,
}

impl Op {
    pub(crate) const fn arity(&self) -> usize {
        match self {
            Self::Copy
            | Self::Reshape { .. }
            | Self::Permute
            | Self::Cont
            | Self::Repeat { .. }
            | Self::Scale(_)
            | Self::SoftMax
            | Self::RMSNorm { .. }
            | Self::RoPE { .. }
            | Self::Silu => 1,
            Self::GetRows | Self::Add | Self::MatMul | Self::Mul => 2,
        }
    }
}
