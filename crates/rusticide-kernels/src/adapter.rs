// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::io::Write;
use std::sync::OnceLock;

#[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
use crate::neon;
use crate::portable;
use crate::q8_input::{Q8Input, Q8Kind};
use crate::quant_kernel::QuantKernel;

pub(crate) type DotRows = fn(&[f32], &[f32], usize, usize, f32, &mut [f32]);
pub(crate) type MixRows = fn(&[f32], &[f32], usize, usize, &mut [f32]);

#[derive(Clone, Copy)]
pub(crate) struct F32Kernels {
    pub(crate) dot: fn(&[f32], &[f32]) -> f32,
    pub(crate) dot_rows: DotRows,
    pub(crate) mix_rows: MixRows,
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
    pub(crate) quantize: fn(&mut Q8Input, &[f32], Q8Kind),
    pub(crate) quantized: &'static [QuantKernel],
}

impl Adapter {
    /// Short stable identifier of the implementation set.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        self.name
    }

    /// The row kernels for the block format with GGML type id `type_id`;
    /// `None` when this adapter has none.
    #[must_use]
    pub fn quant_kernel(&self, type_id: u32) -> Option<&'static QuantKernel> {
        self.quantized
            .iter()
            .find(|kernel| kernel.type_id == type_id)
    }
}

static ACTIVE: OnceLock<&'static Adapter> = OnceLock::new();

#[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
fn compiled_default() -> &'static Adapter {
    &neon::ADAPTER
}

#[cfg(not(all(target_arch = "aarch64", target_feature = "dotprod")))]
fn compiled_default() -> &'static Adapter {
    &portable::ADAPTER
}

/// Resolves the `RUSTICIDE_KERNELS` request against the compiled default;
/// the second value is a warning when the request cannot be honoured.
fn choose(
    requested: Option<&str>,
    compiled: &'static Adapter,
) -> (&'static Adapter, Option<String>) {
    match requested {
        None => (compiled, None),
        Some("portable") => (&portable::ADAPTER, None),
        Some("neon") if compiled.name() == "neon" => (compiled, None),
        Some("neon") => (
            &portable::ADAPTER,
            Some("RUSTICIDE_KERNELS=neon: adapter not compiled in, using portable".to_owned()),
        ),
        Some(other) => (
            compiled,
            Some(format!(
                "RUSTICIDE_KERNELS={other}: unknown adapter, using {}",
                compiled.name()
            )),
        ),
    }
}

/// The adapter every kernel call routes through, chosen on first use from
/// the build target and the `RUSTICIDE_KERNELS` environment variable
/// (`portable` or `neon`).
#[must_use]
pub fn active() -> &'static Adapter {
    ACTIVE.get_or_init(|| {
        let requested = std::env::var("RUSTICIDE_KERNELS").ok();
        select(
            requested.as_deref(),
            compiled_default(),
            &mut std::io::stderr(),
        )
    })
}

/// [`choose`], writing any warning as one line to `diagnostics`; a failed
/// write is dropped because the warning is advisory.
fn select(
    requested: Option<&str>,
    compiled: &'static Adapter,
    diagnostics: &mut impl Write,
) -> &'static Adapter {
    let (adapter, warning) = choose(requested, compiled);
    if let Some(message) = warning {
        let _ = writeln!(diagnostics, "{message}");
    }
    adapter
}

#[cfg(test)]
#[path = "adapter_tests.rs"]
mod tests;
