// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Proof tokens: a value exists only after run-time detection found the
//! instruction set, and every kernel entry point takes one.

/// Proof that this CPU runs NEON and `dotprod`: only [`Dotprod::detect`]
/// makes one. NEON is baseline on `aarch64`, so detection asks for
/// `dotprod` alone.
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy, Debug)]
pub struct Dotprod(());

#[cfg(target_arch = "aarch64")]
impl Dotprod {
    /// A token when this CPU has `dotprod`; std caches the answer.
    #[must_use]
    pub fn detect() -> Option<Self> {
        Self::from_detected(dotprod_detected())
    }

    const fn from_detected(detected: bool) -> Option<Self> {
        if detected { Some(Self(())) } else { None }
    }
}

#[cfg(all(target_arch = "aarch64", not(miri)))]
fn dotprod_detected() -> bool {
    std::arch::is_aarch64_feature_detected!("dotprod")
}

// Miri runs the scalar stand-ins, which need no hardware.
#[cfg(all(target_arch = "aarch64", miri))]
const fn dotprod_detected() -> bool {
    true
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::Dotprod;

    #[test]
    fn from_detected_true_yields_a_token() {
        assert!(Dotprod::from_detected(true).is_some());
    }

    #[test]
    fn from_detected_false_yields_none() {
        assert!(Dotprod::from_detected(false).is_none());
    }

    #[test]
    fn detect_matches_std() {
        let expected = cfg!(miri) || std::arch::is_aarch64_feature_detected!("dotprod");
        assert_eq!(Dotprod::detect().is_some(), expected);
    }
}
