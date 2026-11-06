// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Helpers for the scalar kernel bodies: Miri's stand-ins for the SIMD
//! bodies, and the references the SIMD bodies are tested against.

#[expect(
    clippy::cast_precision_loss,
    reason = "a block sum past 2^24 loses low bits, as the SIMD conversions do"
)]
pub(crate) const fn lossy(value: i32) -> f32 {
    value as f32
}

/// An IEEE half-precision value widened exactly; NaN stays NaN.
pub(crate) fn f16_to_f32(bits: u16) -> f32 {
    let magnitude = u32::from(bits & 0x7fff);
    let shifted = magnitude << 13;
    let widened = if magnitude >= 0x7c00 {
        shifted | 0x7f80_0000
    } else {
        // Rebiasing the exponent by 2^112 also normalizes a subnormal half.
        (f32::from_bits(shifted) * f32::from_bits(0x7780_0000)).to_bits()
    };
    f32::from_bits(widened | (u32::from(bits & 0x8000) << 16))
}

/// The little-endian half at `offset` of `bytes`, widened.
pub(crate) fn f16_at(bytes: &[u8], offset: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]))
}

#[cfg(test)]
mod tests {
    use super::f16_to_f32;

    // `half` converts with inline assembly on this target, which Miri cannot run.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn f16_widening_matches_half_for_every_value() {
        for bits in 0..=u16::MAX {
            let expected = half::f16::from_bits(bits).to_f32();
            let got = f16_to_f32(bits);
            assert!(
                got.to_bits() == expected.to_bits() || (got.is_nan() && expected.is_nan()),
                "{bits:#06x}"
            );
        }
    }
}
