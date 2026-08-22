// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

/// Number of bits at the bottom of a `DispatchKeySet` reserved for
/// backends. Kept at 16 so 48 bits remain for functionality.
pub const NUM_BACKENDS: u8 = 16;

// This value is load-bearing for DispatchKeySet's bit split between backends
// and functionalities. Changing it requires updating Task 8's bit masks and
// the number of variant discriminants in DispatchKey.
const _: () = assert!(NUM_BACKENDS == 16);

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
    /// Whether each backend must supply its own kernel for this functionality.
    ///
    /// Per-backend functionalities (`Dense`, `Autograd`) claim one kernel-table
    /// slot per backend, because each backend has genuinely different logic --
    /// CPU addition and GPU addition are different kernels.
    ///
    /// Cross-cutting functionalities (`Functionalize`) claim exactly one slot
    /// regardless of backend, because they transform the computation graph
    /// before deferring to the backend underneath.
    ///
    /// When adding a new `DispatchKey` variant, decide: does each backend need
    /// its own implementation, or is there one implementation wrapping all?
    #[must_use]
    pub const fn is_per_backend(self) -> bool {
        matches!(self, Self::Dense | Self::Autograd)
    }
}

#[cfg(test)]
mod tests {
    use super::{BackendComponent, DispatchKey};

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
