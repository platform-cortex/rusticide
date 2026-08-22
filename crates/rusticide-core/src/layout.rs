// SPDX-FileCopyrightText: 2026 Rusticide contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

/// How elements are arranged in storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Layout {
    #[default]
    Strided,
}

/// The stride ordering a tensor is expected to hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MemoryFormat {
    #[default]
    Contiguous,
    ChannelsLast,
}
