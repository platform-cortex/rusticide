// SPDX-FileCopyrightText: 2026 Rusticide contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

/// The class of processor a tensor lives on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    Cpu,
    Gpu,
}

/// A specific device: a kind plus an ordinal within that kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Device {
    kind: DeviceKind,
    index: u8,
}

impl Device {
    #[must_use]
    pub const fn cpu() -> Self {
        Self { kind: DeviceKind::Cpu, index: 0 }
    }

    #[must_use]
    pub const fn gpu(index: u8) -> Self {
        Self { kind: DeviceKind::Gpu, index }
    }

    #[must_use]
    pub const fn kind(self) -> DeviceKind {
        self.kind
    }

    #[must_use]
    pub const fn index(self) -> u8 {
        self.index
    }
}

impl Default for Device {
    fn default() -> Self {
        Self::cpu()
    }
}

#[cfg(test)]
mod tests {
    use super::{Device, DeviceKind};

    #[test]
    fn cpu_has_no_index() {
        let d = Device::cpu();
        assert_eq!(d.kind(), DeviceKind::Cpu);
        assert_eq!(d.index(), 0);
    }

    #[test]
    fn gpu_carries_an_index() {
        let d = Device::gpu(2);
        assert_eq!(d.kind(), DeviceKind::Gpu);
        assert_eq!(d.index(), 2);
    }

    #[test]
    fn devices_of_the_same_kind_and_index_are_equal() {
        assert_eq!(Device::gpu(0), Device::gpu(0));
        assert_ne!(Device::gpu(0), Device::gpu(1));
        assert_ne!(Device::cpu(), Device::gpu(0));
    }
}
