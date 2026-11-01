// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::Duration;

use super::kernel_metrics;
use crate::Plan;

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
