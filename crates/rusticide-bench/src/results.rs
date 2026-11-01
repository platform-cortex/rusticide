// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::fmt::Write;

use serde::{Deserialize, Serialize};

use crate::Sample;

/// Measured metrics for one machine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Results {
    pub machine: String,
    pub metrics: BTreeMap<String, Sample>,
}

impl Results {
    /// One table row per metric: name, median and spread in percent.
    #[must_use]
    pub fn to_markdown(&self) -> String {
        let mut table = String::from("| metric | median | spread |\n| --- | --- | --- |\n");
        for (name, sample) in &self.metrics {
            let _ = writeln!(
                table,
                "| {name} | {:.2} | {:.1}% |",
                sample.median,
                sample.spread * 100.0
            );
        }
        table
    }
}

#[cfg(test)]
#[path = "results_tests.rs"]
mod tests;
