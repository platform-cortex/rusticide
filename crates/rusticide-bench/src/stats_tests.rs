// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::Duration;

use super::{Sample, summarize, throughput};
use crate::BenchError;

#[test]
fn summarize_takes_the_median_and_relative_spread() {
    let sample = summarize(&[9.0, 10.0, 12.0]).unwrap();
    assert_eq!(
        sample,
        Sample {
            median: 10.0,
            spread: 0.3
        }
    );
}

#[test]
fn summarize_averages_the_middle_pair_of_an_even_count() {
    let sample = summarize(&[4.0, 2.0, 8.0, 6.0]).unwrap();
    assert_eq!(
        sample,
        Sample {
            median: 5.0,
            spread: 1.2
        }
    );
}

#[test]
fn summarize_of_no_runs_is_an_error() {
    assert!(matches!(summarize(&[]), Err(BenchError::NoRuns)));
}

#[test]
fn throughput_divides_work_by_seconds() {
    assert_eq!(throughput(10.0, Duration::from_millis(500)).unwrap(), 20.0);
}

#[test]
fn throughput_rejects_zero_elapsed_and_zero_work() {
    assert!(matches!(
        throughput(1.0, Duration::ZERO),
        Err(BenchError::ZeroElapsed)
    ));
    assert!(matches!(
        throughput(0.0, Duration::from_secs(1)),
        Err(BenchError::ZeroWork)
    ));
}

#[test]
fn throughput_rejects_non_finite_work_and_overflow() {
    let second = Duration::from_secs(1);
    assert!(matches!(
        throughput(f64::NAN, second),
        Err(BenchError::NotFinite)
    ));
    assert!(matches!(
        throughput(f64::INFINITY, second),
        Err(BenchError::NotFinite)
    ));
    assert!(matches!(
        throughput(f64::MAX, Duration::from_nanos(1)),
        Err(BenchError::NotFinite)
    ));
}

#[test]
fn summarize_rejects_zero_nan_and_negative_runs() {
    for bad in [0.0, f64::NAN, -1.0, f64::INFINITY] {
        assert!(matches!(summarize(&[5.0, bad]), Err(BenchError::NotFinite)));
    }
}
