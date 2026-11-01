// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::Duration;

use rusticide_quantize::block_format;

use super::{kernel_metrics, layout, require_kernel};
use crate::{BenchError, Plan};

#[test]
fn kernel_metrics_cover_every_format_and_both_activations() {
    let plan = Plan {
        runs: 2,
        budget: Duration::from_millis(1),
    };
    let metrics = kernel_metrics(&plan).unwrap();
    let mut keys: Vec<&str> = metrics.keys().map(String::as_str).collect();
    keys.sort_unstable();
    let mut expected: Vec<String> = ["Q4_0", "Q5_0", "Q8_0", "Q4_K", "Q6_K"]
        .iter()
        .flat_map(|name| [format!("kernel/{name}/row"), format!("kernel/{name}/rows8")])
        .chain(["kernel/q8/block".into(), "kernel/q8/superblock".into()])
        .collect();
    expected.sort_unstable();
    assert_eq!(keys, expected);
    assert!(
        metrics
            .values()
            .all(|s| s.median.is_finite() && s.median > 0.0)
    );
}

#[test]
fn a_row_width_that_splits_blocks_is_an_error() {
    let error = layout(block_format(12).unwrap(), 100).unwrap_err();
    assert!(matches!(
        error,
        BenchError::UnalignedRow {
            format: "Q4_K",
            columns: 100
        }
    ));
    assert_eq!(
        error.to_string(),
        "format Q4_K cannot hold rows of 100 columns"
    );
}

#[test]
fn a_format_without_a_kernel_is_an_error() {
    let error = require_kernel(block_format(2).unwrap(), None).unwrap_err();
    assert!(matches!(
        error,
        BenchError::MissingKernel { format: "Q4_0" }
    ));
    assert_eq!(error.to_string(), "no kernel for format Q4_0");
}
