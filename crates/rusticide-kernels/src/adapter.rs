// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::OnceLock;

use crate::portable;

pub(crate) type DotRows = fn(&[f32], &[f32], usize, usize, f32, &mut [f32]);

#[derive(Clone, Copy)]
pub(crate) struct F32Kernels {
    pub(crate) dot: fn(&[f32], &[f32]) -> f32,
    pub(crate) dot_rows: DotRows,
    pub(crate) add_assign: fn(&mut [f32], &[f32]),
    pub(crate) mul_assign: fn(&mut [f32], &[f32]),
    pub(crate) silu_assign: fn(&mut [f32]),
    pub(crate) rms_norm: fn(&mut [f32], f32),
    pub(crate) softmax: fn(&mut [f32]),
    pub(crate) rope_adjacent: fn(&mut [f32], usize, f32, usize),
    pub(crate) rope_half_split: fn(&mut [f32], usize, f32, usize),
}

/// One complete set of kernel implementations, chosen once per process.
pub struct Adapter {
    pub(crate) name: &'static str,
    pub(crate) f32: F32Kernels,
}

impl Adapter {
    /// Short stable identifier of the implementation set.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        self.name
    }
}

static ACTIVE: OnceLock<&'static Adapter> = OnceLock::new();

/// The adapter every kernel call routes through, chosen on first use.
#[must_use]
pub fn active() -> &'static Adapter {
    ACTIVE.get_or_init(|| &portable::ADAPTER)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_adapter_is_portable() {
        assert_eq!(active().name(), "portable");
        assert!(std::ptr::eq(active(), active()));
    }
}
