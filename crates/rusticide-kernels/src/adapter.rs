// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::io::Write;
use std::sync::OnceLock;

#[cfg(target_arch = "aarch64")]
use rusticide_simd::Dotprod;

#[cfg(target_arch = "aarch64")]
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
    /// `None` when this adapter has none. The kernel reads only data
    /// `prepare_matrix` returned it for: row-major only when the kernel has
    /// no layout.
    #[must_use]
    pub fn quant_kernel(&self, type_id: u32) -> Option<&'static QuantKernel> {
        self.quantized
            .iter()
            .find(|kernel| kernel.type_id == type_id)
    }

    /// Repacks a row-major matrix in place for this adapter's kernel for
    /// `type_id` and returns that kernel; when it has none or its layout
    /// does not fit the matrix, the data stays row-major and the `portable`
    /// kernel reads it. `None` when neither has a kernel for the format.
    #[must_use]
    pub fn prepare_matrix(
        &self,
        type_id: u32,
        data: &mut [u8],
        row_bytes: usize,
    ) -> Option<&'static QuantKernel> {
        let own = self.quant_kernel(type_id);
        own.filter(|kernel| {
            kernel
                .layout
                .is_none_or(|layout| (layout.prepare)(data, row_bytes))
        })
        .or_else(|| portable::ADAPTER.quant_kernel(type_id))
    }
}

static ACTIVE: OnceLock<&'static Adapter> = OnceLock::new();

/// Every adapter name `RUSTICIDE_KERNELS` accepts on some CPU.
const KNOWN: [&str; 3] = ["portable", "neon", "avx2"];

/// The adapters this CPU runs, best first and `portable` last; detecting
/// a SIMD adapter stores the proof its kernels read.
pub(crate) fn available() -> Vec<&'static Adapter> {
    ranked(&detected())
}

#[cfg(target_arch = "aarch64")]
fn detected() -> [Option<&'static Adapter>; 1] {
    [Dotprod::detect().map(neon::armed)]
}

#[cfg(not(target_arch = "aarch64"))]
const fn detected() -> [Option<&'static Adapter>; 0] {
    []
}

/// The detected SIMD adapters, best first, then `portable`.
fn ranked(detected: &[Option<&'static Adapter>]) -> Vec<&'static Adapter> {
    detected
        .iter()
        .flatten()
        .copied()
        .chain([&portable::ADAPTER])
        .collect()
}

/// Resolves the `RUSTICIDE_KERNELS` request against the adapters this CPU
/// runs, best first; the second value is a warning when the request
/// cannot be honoured.
fn choose(
    requested: Option<&str>,
    available: &[&'static Adapter],
) -> (&'static Adapter, Option<String>) {
    let best = available.first().copied().unwrap_or(&portable::ADAPTER);
    let Some(name) = requested else {
        return (best, None);
    };
    let mut runnable = available.iter().copied().chain([&portable::ADAPTER]);
    if let Some(adapter) = runnable.find(|adapter| adapter.name() == name) {
        return (adapter, None);
    }
    let reason = if KNOWN.contains(&name) {
        "not supported by this CPU"
    } else {
        "unknown adapter"
    };
    let warning = format!("RUSTICIDE_KERNELS={name}: {reason}, using {}", best.name());
    (best, Some(warning))
}

/// The adapter every kernel call routes through, chosen on first use from
/// what this CPU runs and the `RUSTICIDE_KERNELS` environment variable
/// (`portable`, `neon` or `avx2`).
#[must_use]
pub fn active() -> &'static Adapter {
    ACTIVE.get_or_init(|| {
        let requested = std::env::var("RUSTICIDE_KERNELS").ok();
        select(requested.as_deref(), &available(), &mut std::io::stderr())
    })
}

/// [`choose`], writing any warning as one line to `diagnostics`; a failed
/// write is dropped because the warning is advisory.
fn select(
    requested: Option<&str>,
    available: &[&'static Adapter],
    diagnostics: &mut impl Write,
) -> &'static Adapter {
    let (adapter, warning) = choose(requested, available);
    if let Some(message) = warning {
        let _ = writeln!(diagnostics, "{message}");
    }
    adapter
}

#[cfg(test)]
#[path = "adapter_tests.rs"]
mod tests;
