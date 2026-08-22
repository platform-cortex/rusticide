// SPDX-FileCopyrightText: 2026 Rusticide contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

/// Number of bits at the bottom of a `DispatchKeySet` reserved for
/// backends. Kept at 16 so 48 bits remain for functionality.
pub const NUM_BACKENDS: u8 = 16;

/// A backend that can supply kernels.
///
/// Discriminants start at 1: index 0 means "no backend bit set".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BackendComponent {
    Cpu = 1,
    Gpu = 2,
}

/// A functionality layer.
///
/// **The numeric order of these variants is the dispatch priority
/// specification.** Reordering them silently changes which kernel runs.
/// Higher discriminant wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DispatchKey {
    Dense = 0,
    Functionalize = 1,
    Autograd = 2,
}

impl DispatchKey {
    /// Whether this functionality can be customised per backend.
    #[must_use]
    pub const fn is_per_backend(self) -> bool {
        matches!(self, Self::Dense | Self::Autograd)
    }
}

#[cfg(test)]
mod tests {
    use super::{BackendComponent, DispatchKey, NUM_BACKENDS};

    #[test]
    fn backends_are_numbered_from_one() {
        assert_eq!(BackendComponent::Cpu as u8, 1);
        assert_eq!(BackendComponent::Gpu as u8, 2);
    }

    #[test]
    fn functionality_order_encodes_priority() {
        assert!(
            (DispatchKey::Autograd as u8) > (DispatchKey::Dense as u8),
            "autograd must outrank dense; the enum order IS the spec"
        );
    }

    #[test]
    fn only_some_functionalities_are_per_backend() {
        assert!(DispatchKey::Dense.is_per_backend());
        assert!(DispatchKey::Autograd.is_per_backend());
        assert!(!DispatchKey::Functionalize.is_per_backend());
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn backend_count_leaves_room_for_functionality_bits() {
        assert!(NUM_BACKENDS <= 16);
    }

    #[test]
    fn dispatch_key_discriminants_are_contiguous() {
        // Verify that DispatchKey variants are numbered 0, 1, 2, ...
        // without gaps. Task 8 maps bit index to variant via match on
        // discriminant; a gap or reordering silently produces None and
        // a wrong kernel.
        let dense = DispatchKey::Dense as u8;
        let functionalize = DispatchKey::Functionalize as u8;
        let autograd = DispatchKey::Autograd as u8;

        assert_eq!(dense, 0, "Dense must be 0");
        assert_eq!(functionalize, 1, "Functionalize must be 1");
        assert_eq!(autograd, 2, "Autograd must be 2");

        // Assert strictly increasing order with no gaps
        assert!(functionalize > dense);
        assert!(autograd > functionalize);
        assert_eq!(functionalize, dense + 1);
        assert_eq!(autograd, functionalize + 1);
    }
}
