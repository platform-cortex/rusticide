// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#[inline]
fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

pub(super) fn silu_assign(values: &mut [f32]) {
    for value in values {
        *value = silu(*value);
    }
}

/// In-place `max(x, 0)^2`; negative and `-0.0` inputs give `+0.0`.
pub(super) fn relu_squared_assign(values: &mut [f32]) {
    for value in values {
        let positive = value.max(0.0);
        *value = positive * positive;
    }
}

/// Element-wise `target *= factor`; the shorter slice bounds the work.
pub(super) fn mul_assign(target: &mut [f32], factor: &[f32]) {
    for (slot, &by) in target.iter_mut().zip(factor) {
        *slot *= by;
    }
}

/// Element-wise `target += addend`; the shorter slice bounds the work.
pub(super) fn add_assign(target: &mut [f32], addend: &[f32]) {
    for (slot, &by) in target.iter_mut().zip(addend) {
        *slot += by;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silu_assign_of_zero_is_zero() {
        let mut values = [0.0];
        silu_assign(&mut values);
        assert_eq!(values, [0.0]);
    }

    #[test]
    fn silu_assign_matches_the_definition() {
        let expected = 2.0_f32 / (1.0 + (-2.0_f32).exp());
        let mut values = [2.0, 2.0];
        silu_assign(&mut values);
        assert_eq!(values, [expected, expected]);
    }

    #[test]
    fn relu_squared_assign_squares_positives_and_zeroes_the_rest() {
        let mut values = [-2.0, -0.0, 0.0, 1.5, 3.0];
        relu_squared_assign(&mut values);
        assert_eq!(values, [0.0, 0.0, 0.0, 2.25, 9.0]);
        for value in &values[..3] {
            assert_eq!(value.to_bits(), 0.0_f32.to_bits());
        }
    }

    #[test]
    fn mul_assign_and_add_assign_are_elementwise() {
        let mut target = [1.0, 2.0];
        mul_assign(&mut target, &[3.0, 4.0]);
        assert_eq!(target, [3.0, 8.0]);
        add_assign(&mut target, &[1.0, 1.0]);
        assert_eq!(target, [4.0, 9.0]);
    }
}
