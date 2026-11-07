// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The tests every SIMD body of a format runs, written once for every
//! architecture.

/// The `Format` checks, as tests of the calling module, for `$group` (a `*_group` kernel)
/// under the test host's token, against the scalar body, on the units
/// `$samples::format` makes, seeded from `$seed`. Each `$extra` is a
/// further check of `$samples` taking the format.
macro_rules! group_body_tests {
    ($group:path, $samples:path, $seed:literal $(, $extra:ident)* $(,)?) => {
        use $samples as samples;

        macro_rules! checks {
            () => {
                samples::format(|group, inputs, out| {
                    $group($crate::test_support::token(), group, inputs, out);
                })
            };
        }

        #[test]
        fn the_simd_body_matches_the_scalar_body() {
            checks!().the_simd_body_matches_the_scalar_body($seed);
        }

        #[test]
        fn every_token_is_computed_as_if_alone() {
            checks!().every_token_is_computed_as_if_alone($seed + 1);
        }

        #[test]
        fn the_shorter_of_group_and_input_bounds_the_blocks() {
            checks!().the_shorter_of_group_and_input_bounds_the_blocks($seed + 2);
        }

        #[test]
        fn special_scales_match_the_scalar_body() {
            checks!().special_scales_match_the_scalar_body($seed + 3);
        }

        #[test]
        fn a_group_at_an_odd_address_gives_the_same_products() {
            checks!().a_group_at_an_odd_address_gives_the_same_products($seed + 4);
        }

        #[test]
        fn extreme_bytes_match_the_scalar_body() {
            checks!().extreme_bytes_match_the_scalar_body();
        }

        $(
            #[test]
            fn $extra() {
                samples::$extra(&checks!());
            }
        )*
    };
}

pub(crate) use group_body_tests;
