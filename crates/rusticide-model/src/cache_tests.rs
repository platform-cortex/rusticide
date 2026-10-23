// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::KvCache;
use crate::error::ModelError;

#[test]
fn cache_appends_per_layer() {
    let mut cache = KvCache::new(2, 2, 4);
    cache.store(0, 0, &[1.0, 2.0], &[3.0, 4.0]).unwrap();
    cache.store(0, 1, &[5.0, 6.0], &[7.0, 8.0]).unwrap();
    assert_eq!(cache.keys(0), [1.0, 2.0, 5.0, 6.0]);
    assert_eq!(cache.values(0), [3.0, 4.0, 7.0, 8.0]);
    assert_eq!(cache.keys(1), [] as [f32; 0]);
    assert_eq!(cache.values(1), [] as [f32; 0]);
    assert_eq!(cache.keys(9), [] as [f32; 0]);
    assert_eq!(cache.values(9), [] as [f32; 0]);
}

#[test]
fn cache_rejects_a_skipped_position() {
    let mut cache = KvCache::new(1, 1, 4);
    assert!(matches!(
        cache.store(0, 1, &[1.0], &[1.0]),
        Err(ModelError::CachePosition {
            expected: 0,
            got: 1
        })
    ));
}

#[test]
fn cache_rejects_a_full_context() {
    let mut cache = KvCache::new(1, 1, 1);
    cache.store(0, 0, &[1.0], &[1.0]).unwrap();
    assert!(matches!(
        cache.store(0, 1, &[1.0], &[1.0]),
        Err(ModelError::ContextFull { limit: 1 })
    ));
}

#[test]
fn cache_rejects_an_unknown_layer() {
    let mut cache = KvCache::new(1, 1, 1);
    assert!(matches!(
        cache.store(1, 0, &[1.0], &[1.0]),
        Err(ModelError::InvalidConfig {
            reason: "layer out of range"
        })
    ));
}

#[test]
fn cache_rejects_a_wrong_width() {
    let mut cache = KvCache::new(1, 2, 2);
    assert!(matches!(
        cache.store(0, 0, &[1.0], &[1.0, 2.0]),
        Err(ModelError::InvalidConfig {
            reason: "key or value width mismatch"
        })
    ));
    assert!(matches!(
        cache.store(0, 0, &[1.0, 2.0], &[1.0]),
        Err(ModelError::InvalidConfig {
            reason: "key or value width mismatch"
        })
    ));
}

#[test]
fn cache_clear_empties_every_layer() {
    let mut cache = KvCache::new(2, 1, 2);
    cache.store(0, 0, &[1.0], &[1.0]).unwrap();
    cache.store(1, 0, &[1.0], &[1.0]).unwrap();
    cache.clear();
    assert_eq!(cache.keys(0), [] as [f32; 0]);
    assert_eq!(cache.values(1), [] as [f32; 0]);
    cache.store(0, 0, &[2.0], &[2.0]).unwrap();
    assert_eq!(cache.keys(0), [2.0]);
}
