// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::merge_ids;
use crate::tokenizer::MergeLookup;

/// The quadratic merge `merge_ids` replaced: lowest rank first, leftmost on
/// ties, one pass over every pair per merge.
fn naive(mut ids: Vec<u32>, lookup: &MergeLookup) -> Vec<u32> {
    loop {
        let best = ids
            .windows(2)
            .enumerate()
            .filter_map(|(position, pair)| {
                let &(rank, merged) = lookup.get(&(pair[0], pair[1]))?;
                Some((rank, position, merged))
            })
            .min();
        let Some((_, position, merged)) = best else {
            return ids;
        };
        ids[position] = merged;
        ids.remove(position + 1);
    }
}

/// A deterministic linear congruential generator.
struct Lcg(u64);

impl Lcg {
    fn below(&mut self, bound: u32) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        u32::try_from(self.0 >> 33).unwrap() % bound
    }
}

/// Merges over ids below `symbols + created`, each producing a new id, or now
/// and then an id that already exists; ranks repeat across pairs.
fn random_merges(rng: &mut Lcg, symbols: u32) -> MergeLookup {
    let mut lookup = MergeLookup::new();
    let created = rng.below(12);
    for step in 0..created {
        let pool = symbols + step;
        let pair = (rng.below(pool), rng.below(pool));
        let merged = if rng.below(5) == 0 {
            rng.below(pool)
        } else {
            symbols + step
        };
        let rank = usize::try_from(rng.below(8)).unwrap();
        lookup.insert(pair, (rank, merged));
    }
    lookup
}

/// Random ids, a repeated short pattern, or one id repeated.
fn random_ids(rng: &mut Lcg, symbols: u32) -> Vec<u32> {
    let len = usize::try_from(rng.below(48)).unwrap();
    match rng.below(3) {
        0 => (0..len).map(|_| rng.below(symbols)).collect(),
        1 => {
            let pattern: Vec<u32> = (0..=rng.below(3)).map(|_| rng.below(symbols)).collect();
            pattern.iter().copied().cycle().take(len).collect()
        }
        _ => vec![rng.below(symbols); len],
    }
}

#[test]
fn the_heap_merge_matches_the_naive_merge() {
    let mut rng = Lcg(7);
    for _ in 0..2000 {
        let symbols = 1 + rng.below(4);
        let lookup = random_merges(&mut rng, symbols);
        let ids = random_ids(&mut rng, symbols);
        assert_eq!(merge_ids(ids.clone(), &lookup), naive(ids, &lookup));
    }
}

#[test]
fn a_long_run_merges_like_the_naive_merge() {
    let lookup = MergeLookup::from([((0, 0), (0, 1)), ((1, 1), (1, 2)), ((2, 0), (2, 3))]);
    for len in [0, 1, 2, 3, 511, 512, 513] {
        let ids = vec![0; len];
        assert_eq!(merge_ids(ids.clone(), &lookup), naive(ids, &lookup));
    }
}
