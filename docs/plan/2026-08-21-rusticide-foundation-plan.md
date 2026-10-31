<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Rusticide Foundation (SP0 + SP1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the reset repository to a green build under the
`rusticide` name, with a working tensor core, a schema parser that
round-trips all 2,585 PyTorch operator schemas byte-exactly, and `add`
dispatching through the real dispatch table to a CPU kernel.

**Architecture:** Bottom-up. Everything is blocked on a `Tensor` type, so
`rusticide-core` is built first from its leaf value types (`DType`,
`Device`, `Shape`) upward through `Storage` and the dispatch-key algebra.
`rusticide-schema` is independent of the core and can be built in
parallel. `rusticide-dispatch` sits on both. The surviving `rusticide-*`
crates are rewired last, once the types they import exist again.

**Tech Stack:** Rust 1.99 toolchain, MSRV 1.96, edition 2024, resolver 3.
`thiserror` for errors, `rayon` for parallelism, `half` for f16/bf16.
No C, no C++, no Python, no CPython.

**Spec:** `docs/plan/2026-08-21-rusticide-framework-design.md`

## Global Constraints

Every task's requirements implicitly include this section.

- **Pure Rust.** No C, C++, Python, or CPython. No `pyo3`, no `libtorch`,
  no `cc`/`bindgen` of vendored C, no BLAS, no cuDNN, no NCCL.
- **`unsafe_code = "deny"`** workspace-wide. Only `rusticide-alloc` may
  opt out in this plan, under the policy in the spec: isolation, safe
  public API, `// SAFETY:` per block, README justification, an entry in
  `docs/UNSAFE.md`, and Miri in CI.
- **No `unwrap`/`expect` in production code.** `clippy::unwrap_used` and
  `clippy::expect_used` are warnings, escalated to errors by
  `-D warnings`. Both are allowed inside `#[cfg(test)]` via
  `.clippy.toml`.
- **`-D warnings`** in `.cargo/config.toml`. Every rustc and clippy
  warning fails the build. Clippy runs `pedantic`, `nursery`, and `cargo`
  groups.
- **Files under 300 lines**, tests included. Split by responsibility and
  re-export from the crate root.
- **ASCII only** in new code comments.
- **SPDX header** on every new `.rs` file, as a fenced block below rather
  than inline code -- `comply` scans this document and reads a trailing
  backtick as part of the licence identifier:

  ```text
  // SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
  //
  // SPDX-License-Identifier: MIT OR Apache-2.0
  ```

  Three lines, with the bare `//` separator. This is the form `REUSE.toml`
  declares and the form every committed source file uses; an earlier
  version of this plan said "2026 Rusticide contributors" on two lines,
  which is wrong on both counts.
- **Commit messages** follow `committed.toml` and `commit-types.toml`.
  Commits are GPG-signed: use `git commit -S`.
- **No attribution trailers of any kind.** Not `Co-authored-by:`, and
  not `Assisted-by:` either -- `AGENTS.md` line 209 suggests the latter,
  and the project owner has overridden it.
- **Never bypass the push guard.** `.claude/settings.json` denies
  `git push` and `gh pr create`; publishing stays a human action.

**Every task runs `cargo fmt` before committing, and `cargo fmt --check`
is part of its verification.** The `hk` pre-commit hook that would
normally enforce this is broken in the current environment and is
bypassed with `HK=0`, so formatting has to be checked explicitly. Tasks
2 through 4 shipped unformatted because this was missed; do not repeat
it. Note that the code blocks in this plan are hand-written and are NOT
rustfmt-normalized -- transcribe them, then run `cargo fmt`.

Verification commands used throughout:

```bash
mise run check      # cargo check across the workspace
mise run clippy     # clippy with the workspace lint config
mise run test       # cargo test
mise run fmt        # rustfmt
mise run lint       # fmt:check + clippy + taplo + rumdl + actionlint
mise run pre-commit # the full gate before committing
```

---

## File Structure

New crates, each with `Cargo.toml`, `README.md`, `LICENSE-MIT`,
`LICENSE-APACHE-2.0`:

- `crates/rusticide-core/` -- value types and the dispatch-key algebra.
  - `src/lib.rs` -- re-exports only.
  - `src/dtype.rs` -- `DType`.
  - `src/device.rs` -- `Device`, `DeviceKind`.
  - `src/layout.rs` -- `Layout`, `MemoryFormat`.
  - `src/error.rs` -- `Error`, `Result`.
  - `src/shape.rs` -- `Shape`.
  - `src/strides.rs` -- `Strides`, contiguity.
  - `src/storage.rs` -- `Storage`.
  - `src/key/mod.rs` -- re-exports.
  - `src/key/dispatch_key.rs` -- `BackendComponent`, `DispatchKey`.
  - `src/key/key_set.rs` -- `DispatchKeySet`.
  - `src/tensor/mod.rs` -- re-exports.
  - `src/tensor/impl_.rs` -- `TensorImpl`.
  - `src/tensor/handle.rs` -- `Tensor`.
  - `src/tensor/version.rs` -- `VersionCounter`.
- `crates/rusticide-alloc/` -- the only `unsafe`-permitted crate here.
  - `src/lib.rs` -- re-exports.
  - `src/buffer.rs` -- `Buffer`, aligned and uninitialized allocation.
- `crates/rusticide-schema/` -- the operator schema data model.
  - `src/lib.rs` -- re-exports.
  - `src/ty.rs` -- `BaseTy`, `Type`.
  - `src/annotation.rs` -- `Annotation`.
  - `src/argument.rs` -- `Argument`, `Arguments`, `Return`.
  - `src/name.rs` -- `OperatorName`.
  - `src/schema.rs` -- `FunctionSchema`.
  - `tests/corpus.rs` -- the round-trip test over the vendored corpus.
  - `tests/data/native_functions.yaml` -- vendored, 631 KB.
- `crates/rusticide-dispatch/` -- the runtime dispatcher.
  - `src/lib.rs` -- re-exports.
  - `src/ivalue.rs` -- `IValue`.
  - `src/stack.rs` -- `Stack`.
  - `src/kernel.rs` -- `KernelFunction`.
  - `src/aliasing.rs` -- the self-aliasing guard.
  - `src/entry.rs` -- `OperatorEntry`.
  - `src/dispatcher.rs` -- `Dispatcher`.
- `crates/rusticide-io/` -- GGUF reader and tensor materialization.
- `crates/rusticide/` -- facade and CLI.
  - `src/lib.rs`, `src/main.rs`, `src/cli/mod.rs`,
    `src/cli/cmd/{mod,schema,version}.rs`.

Kept in place (no renames): `crates/rusticide-ggml` (legacy compute
graph), `-tokenizer`, `-quantize`, `-sampling` and `-fuzz`.

---

## Task 1: Repair the workspace manifest

The repository was reset: nine crates were deleted and the workspace no
longer resolves. Nothing else can be verified until `cargo metadata`
succeeds.

**Files:**

- Modify: `Cargo.toml`

**Interfaces:**

- Consumes: nothing.
- Produces: a resolvable workspace whose `[workspace.dependencies]` names
  only crates that exist.

**Steps:**

- [ ] **Step 1: Confirm the current failure**

Run: `cargo metadata --no-deps --format-version 1 >/dev/null`
Expected: exit 101, `failed to load manifest for dependency
'rusticide-tensor'`.

- [ ] **Step 2: Point workspace metadata at the new project**

In `Cargo.toml`, replace the `[workspace.package]` values:

```toml
description = "Pure-Rust deep learning framework"
documentation = "https://docs.rs/rusticide"
repository = "https://github.com/platform-cortex/rusticide"
```

- [ ] **Step 3: Drop dependencies on deleted crates**

In `[workspace.dependencies]`, delete the `rusticide`, `rusticide-bitnet`,
`rusticide-context`, `rusticide-model`, `rusticide-tensor`, and
`rusticide-train` lines.
Update `exclude` to `["crates/rusticide-fuzz"]`.

- [ ] **Step 4: Temporarily narrow the members list**

`crates/*` currently globs crates that cannot compile. Replace with an
explicit list so the workspace resolves while the core is built:

```toml
members = ["crates/rusticide-core"]
```

Later tasks add each crate to this list as it becomes buildable. Task 20
restores the `crates/*` glob.

- [ ] **Step 5: Verify**

Run: `cargo metadata --no-deps --format-version 1 >/dev/null`
Expected: exit 0. It will warn that `crates/rusticide-core` does not
exist yet -- create it in Task 2 and re-run.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml
git commit -S -m "build: repair workspace after reset

The reset deleted nine crates; the workspace still named them in
[workspace.dependencies] and globbed them via crates/*, so cargo
metadata failed. Point package metadata at rusticide and narrow
members to what exists."
```

---

## Task 2: rusticide-core -- DType

**Files:**

- Create: `crates/rusticide-core/Cargo.toml`
- Create: `crates/rusticide-core/README.md`
- Create: `crates/rusticide-core/src/lib.rs`
- Create: `crates/rusticide-core/src/dtype.rs`

**Interfaces:**

- Consumes: nothing.
- Produces: `DType` with `size_in_bytes() -> usize`,
  `is_floating_point() -> bool`, `is_signed() -> bool`, and
  `promote(self, other: DType) -> DType`.

**Steps:**

- [ ] **Step 1: Create the crate manifest**

`crates/rusticide-core/Cargo.toml`:

```toml
[package]
authors.workspace = true
categories = ["science", "data-structures"]
description = "Rusticide core value types and dispatch key algebra"
edition.workspace = true
keywords = ["tensor", "deep-learning", "dispatch", "machine-learning"]
license.workspace = true
name = "rusticide-core"
readme = "README.md"
repository.workspace = true
rust-version.workspace = true
version.workspace = true

[dependencies]
half.workspace = true
thiserror.workspace = true

[lints]
workspace = true
```

- [ ] **Step 2: Write the failing test**

`crates/rusticide-core/src/dtype.rs`:

```rust
// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
mod tests {
    use super::DType;

    #[test]
    fn size_in_bytes_matches_the_scalar_width() {
        assert_eq!(DType::F32.size_in_bytes(), 4);
        assert_eq!(DType::F16.size_in_bytes(), 2);
        assert_eq!(DType::I64.size_in_bytes(), 8);
        assert_eq!(DType::Bool.size_in_bytes(), 1);
    }

    #[test]
    fn promotion_picks_the_wider_type() {
        assert_eq!(DType::I32.promote(DType::F32), DType::F32);
        assert_eq!(DType::F32.promote(DType::I32), DType::F32);
        assert_eq!(DType::I32.promote(DType::I64), DType::I64);
        assert_eq!(DType::Bool.promote(DType::I8), DType::I8);
    }

    #[test]
    fn promotion_of_half_widths_yields_the_wider_float() {
        assert_eq!(DType::F16.promote(DType::BF16), DType::F32);
        assert_eq!(DType::F16.promote(DType::F32), DType::F32);
    }
}
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cargo test -p rusticide-core dtype`
Expected: FAIL, `cannot find type 'DType' in this scope`.

- [ ] **Step 4: Implement `DType`**

Above the test module in the same file:

```rust
/// A scalar element type.
///
/// Ordering of the variants is not semantically meaningful; promotion is
/// decided by `promotion_rank`, not by variant order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    U8,
    I8,
    I16,
    I32,
    I64,
    F16,
    BF16,
    F32,
    F64,
}

impl DType {
    #[must_use]
    pub const fn size_in_bytes(self) -> usize {
        match self {
            Self::Bool | Self::U8 | Self::I8 => 1,
            Self::I16 | Self::F16 | Self::BF16 => 2,
            Self::I32 | Self::F32 => 4,
            Self::I64 | Self::F64 => 8,
        }
    }

    #[must_use]
    pub const fn is_floating_point(self) -> bool {
        matches!(self, Self::F16 | Self::BF16 | Self::F32 | Self::F64)
    }

    #[must_use]
    pub const fn is_signed(self) -> bool {
        !matches!(self, Self::Bool | Self::U8)
    }

    /// Rank within a promotion category. Higher wins.
    const fn promotion_rank(self) -> u8 {
        match self {
            Self::Bool => 0,
            Self::U8 | Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 3,
            Self::I64 => 4,
            Self::F16 | Self::BF16 => 5,
            Self::F32 => 6,
            Self::F64 => 7,
        }
    }

    /// The type both operands are cast to before an operation.
    ///
    /// Floating point always beats integral. F16 and BF16 have no
    /// ordering between them, so mixing them widens to F32 rather than
    /// silently picking one.
    #[must_use]
    pub const fn promote(self, other: Self) -> Self {
        if matches!(
            (self, other),
            (Self::F16, Self::BF16) | (Self::BF16, Self::F16)
        ) {
            return Self::F32;
        }
        if self.is_floating_point() == other.is_floating_point() {
            if self.promotion_rank() >= other.promotion_rank() {
                self
            } else {
                other
            }
        } else if self.is_floating_point() {
            self
        } else {
            other
        }
    }
}
```

- [ ] **Step 5: Wire up the crate root**

`crates/rusticide-core/src/lib.rs`:

```rust
// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rusticide core: value types and the dispatch key algebra.

mod dtype;

pub use dtype::DType;
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cargo test -p rusticide-core dtype`
Expected: 3 passed.

- [ ] **Step 7: Verify lints and commit**

```bash
mise run clippy
git add crates/rusticide-core Cargo.toml
git commit -S -m "feat(core): add DType with promotion rules

Mixing F16 and BF16 widens to F32 rather than picking arbitrarily:
neither is representable in the other, so there is no correct choice
between them."
```

---

## Task 3: rusticide-core -- Device, Layout, Error

**Files:**

- Create: `crates/rusticide-core/src/device.rs`
- Create: `crates/rusticide-core/src/layout.rs`
- Create: `crates/rusticide-core/src/error.rs`
- Modify: `crates/rusticide-core/src/lib.rs`

**Interfaces:**

- Consumes: `DType` from Task 2.
- Produces: `Device`, `DeviceKind`, `Layout`, `MemoryFormat`, `Error`,
  and `Result<T> = core::result::Result<T, Error>`.

**Steps:**

- [ ] **Step 1: Write the failing tests**

`crates/rusticide-core/src/device.rs`:

```rust
// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

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
```

`crates/rusticide-core/src/error.rs`:

```rust
// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
mod tests {
    use super::Error;
    use crate::DType;

    #[test]
    fn dtype_mismatch_names_both_types() {
        let e = Error::DTypeMismatch {
            expected: DType::F32,
            got: DType::I64,
        };
        let msg = e.to_string();
        assert!(msg.contains("F32"), "message was: {msg}");
        assert!(msg.contains("I64"), "message was: {msg}");
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p rusticide-core`
Expected: FAIL, unresolved `Device` and `Error`.

- [ ] **Step 3: Implement `Device`**

Above the tests in `device.rs`:

```rust
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
```

- [ ] **Step 4: Implement `Layout`**

`crates/rusticide-core/src/layout.rs`:

```rust
// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
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
```

- [ ] **Step 5: Implement `Error`**

Above the tests in `error.rs`:

```rust
use crate::DType;

/// Errors raised by the core value types.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("dtype mismatch: expected {expected:?}, got {got:?}")]
    DTypeMismatch { expected: DType, got: DType },

    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },

    #[error("shapes {lhs:?} and {rhs:?} are not broadcastable")]
    NotBroadcastable { lhs: Vec<usize>, rhs: Vec<usize> },

    #[error("dimension {dim} is out of range for a {ndim}-D tensor")]
    DimOutOfRange { dim: i64, ndim: usize },

    #[error("allocation of {bytes} bytes failed")]
    AllocationFailed { bytes: usize },
}

/// The crate's result alias.
pub type Result<T> = core::result::Result<T, Error>;
```

- [ ] **Step 6: Re-export from the crate root**

In `lib.rs`, add the modules and re-exports:

```rust
mod device;
mod dtype;
mod error;
mod layout;

pub use device::{Device, DeviceKind};
pub use dtype::DType;
pub use error::{Error, Result};
pub use layout::{Layout, MemoryFormat};
```

- [ ] **Step 7: Run the tests and commit**

Run: `cargo test -p rusticide-core` -- expected: 7 passed.

```bash
mise run clippy
git add crates/rusticide-core
git commit -S -m "feat(core): add Device, Layout and Error types"
```

---

## Task 4: rusticide-core -- Shape and Strides

**Files:**

- Create: `crates/rusticide-core/src/shape.rs`
- Create: `crates/rusticide-core/src/strides.rs`
- Modify: `crates/rusticide-core/src/lib.rs`

**Interfaces:**

- Consumes: `Error`, `Result` from Task 3.
- Produces: `Shape` with `numel()`, `ndim()`, `dims()`,
  `broadcast_with(&self, &Shape) -> Result<Shape>`; `Strides` with
  `contiguous_for(&Shape) -> Strides` and `is_contiguous(&self, &Shape)`.

**Steps:**

- [ ] **Step 1: Write the failing tests**

`crates/rusticide-core/src/shape.rs`:

```rust
// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
mod tests {
    use super::Shape;

    #[test]
    fn numel_is_the_product_of_dims() {
        assert_eq!(Shape::new(&[2, 3, 4]).numel(), 24);
    }

    #[test]
    fn a_scalar_has_zero_dims_and_one_element() {
        let s = Shape::new(&[]);
        assert_eq!(s.ndim(), 0);
        assert_eq!(s.numel(), 1);
    }

    #[test]
    fn a_zero_dim_makes_numel_zero() {
        assert_eq!(Shape::new(&[2, 0, 4]).numel(), 0);
    }

    #[test]
    fn broadcast_stretches_ones_and_pads_on_the_left() {
        let a = Shape::new(&[3, 1, 5]);
        let b = Shape::new(&[4, 5]);
        let r = a.broadcast_with(&b).expect("shapes broadcast");
        assert_eq!(r.dims(), &[3, 4, 5]);
    }

    #[test]
    fn broadcast_rejects_incompatible_dims() {
        let a = Shape::new(&[3, 2]);
        let b = Shape::new(&[3, 4]);
        assert!(a.broadcast_with(&b).is_err());
    }
}
```

`crates/rusticide-core/src/strides.rs`:

```rust
// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
mod tests {
    use super::Strides;
    use crate::Shape;

    #[test]
    fn contiguous_strides_are_row_major() {
        let shape = Shape::new(&[2, 3, 4]);
        assert_eq!(Strides::contiguous_for(&shape).as_slice(), &[12, 4, 1]);
    }

    #[test]
    fn contiguous_strides_recognise_themselves() {
        let shape = Shape::new(&[2, 3, 4]);
        let strides = Strides::contiguous_for(&shape);
        assert!(strides.is_contiguous(&shape));
    }

    #[test]
    fn permuted_strides_are_not_contiguous() {
        let shape = Shape::new(&[2, 3]);
        let strides = Strides::new(&[1, 2]);
        assert!(!strides.is_contiguous(&shape));
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p rusticide-core`
Expected: FAIL, unresolved `Shape` and `Strides`.

- [ ] **Step 3: Implement `Shape`**

```rust
use crate::{Error, Result};

/// The extent of each dimension of a tensor.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    #[must_use]
    pub fn new(dims: &[usize]) -> Self {
        Self { dims: dims.to_vec() }
    }

    #[must_use]
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    #[must_use]
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Total element count. A zero-dimensional shape holds one element.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    /// NumPy broadcasting: align from the right, pad the shorter shape
    /// with implicit leading ones, and stretch any extent of 1.
    pub fn broadcast_with(&self, other: &Self) -> Result<Self> {
        let ndim = self.ndim().max(other.ndim());
        let mut dims = Vec::with_capacity(ndim);
        for i in 0..ndim {
            let a = self.dim_from_right(i);
            let b = other.dim_from_right(i);
            let d = match (a, b) {
                (x, y) if x == y => x,
                (1, y) => y,
                (x, 1) => x,
                _ => {
                    return Err(Error::NotBroadcastable {
                        lhs: self.dims.clone(),
                        rhs: other.dims.clone(),
                    });
                }
            };
            dims.push(d);
        }
        dims.reverse();
        Ok(Self { dims })
    }

    /// Extent `i` positions from the right, or 1 past the left edge.
    fn dim_from_right(&self, i: usize) -> usize {
        if i < self.dims.len() {
            self.dims[self.dims.len() - 1 - i]
        } else {
            1
        }
    }
}
```

- [ ] **Step 4: Implement `Strides`**

```rust
use crate::Shape;

/// The element step along each dimension.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Strides {
    strides: Vec<usize>,
}

impl Strides {
    #[must_use]
    pub fn new(strides: &[usize]) -> Self {
        Self { strides: strides.to_vec() }
    }

    #[must_use]
    pub fn as_slice(&self) -> &[usize] {
        &self.strides
    }

    /// Row-major strides: the last dimension steps by one.
    #[must_use]
    pub fn contiguous_for(shape: &Shape) -> Self {
        let dims = shape.dims();
        let mut strides = vec![1; dims.len()];
        for i in (0..dims.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        Self { strides }
    }

    #[must_use]
    pub fn is_contiguous(&self, shape: &Shape) -> bool {
        self.strides == Self::contiguous_for(shape).strides
    }
}
```

- [ ] **Step 5: Re-export, test and commit**

Add `mod shape; mod strides;` and
`pub use shape::Shape; pub use strides::Strides;` to `lib.rs`.

Run: `cargo test -p rusticide-core` -- expected: 15 passed.

```bash
mise run clippy
git add crates/rusticide-core
git commit -S -m "feat(core): add Shape with broadcasting and Strides"
```

---

## Task 5: rusticide-alloc -- the first unsafe-permitted crate

This task exercises the `unsafe` policy end to end. It is deliberately
small so the policy is proven on something auditable.

**Files:**

- Create: `crates/rusticide-alloc/Cargo.toml`
- Create: `crates/rusticide-alloc/README.md`
- Create: `crates/rusticide-alloc/src/lib.rs`
- Create: `crates/rusticide-alloc/src/buffer.rs`
- Create: `docs/UNSAFE.md`
- Modify: `Cargo.toml`

**Interfaces:**

- Consumes: nothing from other crates.
- Produces: `Buffer` with `alloc_zeroed(bytes: usize) -> Option<Buffer>`,
  `as_slice(&self) -> &[u8]`, `as_mut_slice(&mut self) -> &mut [u8]`,
  and `len(&self) -> usize`.

**Steps:**

- [ ] **Step 1: Create the manifest with its own lint table**

The crate must NOT inherit `[lints] workspace = true`, because the
workspace denies `unsafe_code`.

`crates/rusticide-alloc/Cargo.toml`:

```toml
[package]
authors.workspace = true
categories = ["memory-management"]
description = "Aligned buffer allocation for Rusticide tensors"
edition.workspace = true
keywords = ["allocator", "aligned", "buffer", "memory", "tensor"]
license.workspace = true
name = "rusticide-alloc"
readme = "README.md"
repository.workspace = true
rust-version.workspace = true
version.workspace = true

[dependencies]

[lints.rust]
rust_2018_idioms = { level = "warn", priority = -1 }
unexpected_cfgs = "deny"
unnameable_types = "warn"
unreachable_pub = "warn"
unsafe_code = "allow"
unused_lifetimes = "warn"
unused_macro_rules = "warn"
unused_qualifications = "warn"
warnings = "deny"

[lints.clippy]
cargo = { level = "warn", priority = -1 }
cargo_common_metadata = "allow"
collapsible_if = "allow"
expect_used = "warn"
multiple_crate_versions = "allow"
nursery = { level = "warn", priority = -1 }
pedantic = { level = "warn", priority = -1 }
redundant_pub_crate = "allow"
undocumented_unsafe_blocks = "deny"
unwrap_used = "warn"
```

**This table must be the workspace's, with `unsafe_code` flipped and
`undocumented_unsafe_blocks` added -- nothing else.** Declaring its own
`[lints.rust]` means the crate stops inheriting, so any workspace lint
left out is not merely un-restated: `unnameable_types`,
`unused_macro_rules` and `unused_qualifications` are allow-by-default in
rustc, so omitting them switches them off, and `warnings = "deny"` does
not rescue a lint that never fires. An `unsafe` exemption that quietly
drops unrelated lints is the exact failure this policy exists to prevent.

`redundant_pub_crate` is carried over because the workspace's own comment
records that it contradicts `unreachable_pub`, which this crate does
inherit at `warn`; leaving it out creates a latent unsatisfiable pair.

`clippy::undocumented_unsafe_blocks` mechanically enforces the
`// SAFETY:` requirement from the spec's policy.

- [ ] **Step 2: Write the failing test**

`crates/rusticide-alloc/src/buffer.rs`:

```rust
// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
mod tests {
    use super::Buffer;

    #[test]
    fn zeroed_buffer_has_the_requested_length_and_is_zero() {
        let b = Buffer::alloc_zeroed(64).expect("allocation succeeds");
        assert_eq!(b.len(), 64);
        assert!(b.as_slice().iter().all(|&x| x == 0));
    }

    #[test]
    fn a_zero_length_buffer_is_valid_and_empty() {
        let b = Buffer::alloc_zeroed(0).expect("allocation succeeds");
        assert_eq!(b.len(), 0);
        assert!(b.as_slice().is_empty());
    }

    #[test]
    fn writes_through_the_mutable_slice_are_visible() {
        let mut b = Buffer::alloc_zeroed(4).expect("allocation succeeds");
        b.as_mut_slice()[2] = 7;
        assert_eq!(b.as_slice(), &[0, 0, 7, 0]);
    }
}
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cargo test -p rusticide-alloc`
Expected: FAIL, `cannot find type 'Buffer'`.

- [ ] **Step 4: Implement `Buffer`**

The minimal implementation is safe: `vec![0u8; bytes]` already gives a
zeroed, correctly aligned allocation. YAGNI says start there. The crate
exists because uninitialized and over-aligned allocation will need
`unsafe` in a later sub-project; the lint table and `docs/UNSAFE.md`
entry are established now so the policy is in place before it is needed.

```rust
/// An owned, byte-addressed allocation backing a tensor's storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Buffer {
    bytes: Vec<u8>,
}

impl Buffer {
    /// Allocate `bytes` zero-filled bytes.
    ///
    /// Returns `None` if the allocation fails.
    #[must_use]
    pub fn alloc_zeroed(bytes: usize) -> Option<Self> {
        let mut v = Vec::new();
        v.try_reserve_exact(bytes).ok()?;
        v.resize(bytes, 0);
        Some(Self { bytes: v })
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.bytes
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.bytes
    }
}
```

- [ ] **Step 5: Write `docs/UNSAFE.md`**

```markdown
<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
SPDX-License-Identifier: CC-BY-3.0+
-->

# Crates permitted to use `unsafe`

`unsafe_code = "deny"` is the workspace default. Every crate listed here
opts out with its own `[lints.rust]` table, and every one must satisfy
the policy in `docs/plan/2026-08-21-rusticide-framework-design.md`:
isolation, a safe public API, a `// SAFETY:` comment per block, a README
justification, an entry in this file, and Miri in CI.

If this list stops being short, the policy has failed.

## `rusticide-alloc`

**Capability:** allocating tensor buffers.

**Why safe Rust is not enough:** zero-filling every allocation costs real
time on a framework that immediately overwrites most of them. Avoiding
that needs `MaybeUninit` and the raw allocator API. Over-aligned
allocation for SIMD loads has the same requirement.

**Current state:** the implementation is entirely safe today
(`Vec::try_reserve_exact` plus `resize`). The crate and its lint table
exist so the boundary is in place before the uninitialized path lands.

**How it could be deleted:** if a future Rust release offers safe
uninitialized byte buffers with an allocation-failure result, this crate
folds back into `rusticide-core`.
```

- [ ] **Step 6: Write the crate README**

`crates/rusticide-alloc/README.md` states the same justification in
short form and links to `docs/UNSAFE.md`.

- [ ] **Step 7: Add to the workspace, test, and commit**

Add `"crates/rusticide-alloc"` to `members` in the root `Cargo.toml`.

Run: `cargo test -p rusticide-alloc` -- expected: 3 passed.
Run: `mise run miri` -- expected: pass.

```bash
mise run clippy
git add crates/rusticide-alloc docs/UNSAFE.md Cargo.toml
git commit -S -m "feat(alloc): add Buffer and establish the unsafe policy

The implementation is safe today; the crate exists so the boundary and
docs/UNSAFE.md are in place before the MaybeUninit path is needed."
```

---

## Task 6: rusticide-core -- Storage

Storage is where the spec's aliasing decision becomes code:
`Arc<RwLock<Buffer>>` so parameters can be shared and mutated.

**Files:**

- Create: `crates/rusticide-core/src/storage.rs`
- Modify: `crates/rusticide-core/src/lib.rs`
- Modify: `crates/rusticide-core/Cargo.toml`

**Interfaces:**

- Consumes: `Buffer` (Task 5), `DType` (Task 2), `Device` (Task 3).
- Produces: `Storage` with `new(DType, Device, numel) -> Result<Storage>`,
  `read()`, `write()`, `dtype()`, `device()`, `numel()`, and
  `same_buffer_as(&self, &Storage) -> bool`.

**Steps:**

- [ ] **Step 1: Write the failing test**

```rust
// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
mod tests {
    use super::Storage;
    use crate::{DType, Device};

    #[test]
    fn new_storage_is_sized_from_dtype_and_count() {
        let s = Storage::new(DType::F32, Device::cpu(), 10)
            .expect("allocation succeeds");
        assert_eq!(s.numel(), 10);
        assert_eq!(s.read().len(), 40);
    }

    #[test]
    fn clones_share_one_buffer() {
        let a = Storage::new(DType::U8, Device::cpu(), 4)
            .expect("allocation succeeds");
        let b = a.clone();
        b.write()[0] = 9;
        assert_eq!(a.read()[0], 9, "the clone must alias, not copy");
        assert!(a.same_buffer_as(&b));
    }

    #[test]
    fn distinct_allocations_do_not_alias() {
        let a = Storage::new(DType::U8, Device::cpu(), 4)
            .expect("allocation succeeds");
        let b = Storage::new(DType::U8, Device::cpu(), 4)
            .expect("allocation succeeds");
        assert!(!a.same_buffer_as(&b));
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p rusticide-core storage`
Expected: FAIL, `cannot find type 'Storage'`.

- [ ] **Step 3: Implement `Storage`**

```rust
use alloc::sync::Arc;
use core::sync::atomic::Ordering;
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use rusticide_alloc::Buffer;

use crate::{DType, Device, Error, Result};

/// The buffer backing one or more tensors.
///
/// Cloning a `Storage` aliases the same buffer. That is required, not
/// incidental: a training loop mutates parameters that the module and
/// the optimizer both hold, and both must observe the update. See
/// "Aliasing and mutation" in the design document.
#[derive(Debug, Clone)]
pub struct Storage {
    buffer: Arc<RwLock<Buffer>>,
    dtype: DType,
    device: Device,
    numel: usize,
}

impl Storage {
    pub fn new(dtype: DType, device: Device, numel: usize) -> Result<Self> {
        let bytes = numel
            .checked_mul(dtype.size_in_bytes())
            .ok_or(Error::AllocationFailed { bytes: usize::MAX })?;
        let buffer = Buffer::alloc_zeroed(bytes)
            .ok_or(Error::AllocationFailed { bytes })?;
        Ok(Self {
            buffer: Arc::new(RwLock::new(buffer)),
            dtype,
            device,
            numel,
        })
    }

    /// A shared view of the bytes.
    ///
    /// # Panics
    ///
    /// Panics if the lock is poisoned, which can only happen if another
    /// thread panicked while holding it. There is no meaningful recovery
    /// from a half-written tensor.
    #[must_use]
    pub fn read(&self) -> RwLockReadGuard<'_, Buffer> {
        match self.buffer.read() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    /// An exclusive view of the bytes.
    #[must_use]
    pub fn write(&self) -> RwLockWriteGuard<'_, Buffer> {
        match self.buffer.write() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    #[must_use]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    #[must_use]
    pub const fn device(&self) -> Device {
        self.device
    }

    #[must_use]
    pub const fn numel(&self) -> usize {
        self.numel
    }

    /// Whether two storages are backed by the same allocation.
    ///
    /// The dispatcher uses this to detect self-aliasing calls such as
    /// `a.add_(a)`, which would otherwise deadlock by taking a read and
    /// a write lock on one buffer.
    #[must_use]
    pub fn same_buffer_as(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.buffer, &other.buffer)
    }
}
```

Note the guards return `Buffer`, and `Buffer::as_slice` gives the bytes;
the test calls `s.read().len()` via `Deref`. Add
`impl core::ops::Deref for Buffer { type Target = [u8]; ... }` and
`DerefMut` in `rusticide-alloc/src/buffer.rs` so the guards index
directly. Write the deref tests alongside.

- [ ] **Step 4: Add the dependency**

In `crates/rusticide-core/Cargo.toml`:

```toml
rusticide-alloc.workspace = true
```

and in the root `[workspace.dependencies]`:

```toml
rusticide-alloc = { path = "crates/rusticide-alloc", version = "0.1.0" }
```

- [ ] **Step 5: Run the tests and commit**

Run: `cargo test -p rusticide-core` -- expected: 18 passed.

```bash
mise run clippy
git add crates/rusticide-core crates/rusticide-alloc Cargo.toml
git commit -S -m "feat(core): add Storage with shared mutable buffers

Arc<RwLock<Buffer>> rather than Arc::make_mut: copy-on-write would
clone a parameter on p.add_(grad) and the module would never see the
update. same_buffer_as exists for the dispatcher's self-aliasing guard."
```

---

## Task 7: rusticide-core -- DispatchKey enums

**Files:**

- Create: `crates/rusticide-core/src/key/mod.rs`
- Create: `crates/rusticide-core/src/key/dispatch_key.rs`
- Modify: `crates/rusticide-core/src/lib.rs`

**Interfaces:**

- Consumes: nothing.
- Produces: `BackendComponent` and `DispatchKey` enums, plus
  `NUM_BACKENDS: u8` and `DispatchKey::is_per_backend(self) -> bool`.

**Steps:**

- [ ] **Step 1: Write the failing test**

```rust
// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

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
    fn backend_count_leaves_room_for_functionality_bits() {
        assert!(NUM_BACKENDS <= 16);
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p rusticide-core key`
Expected: FAIL, unresolved names.

- [ ] **Step 3: Implement the enums**

```rust
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
```

- [ ] **Step 4: Run the tests and commit**

Add `mod key;` and `pub use key::{BackendComponent, DispatchKey,
NUM_BACKENDS};` to `lib.rs`, with `key/mod.rs` re-exporting from
`dispatch_key.rs`.

Run: `cargo test -p rusticide-core` -- expected: 22 passed.

```bash
git commit -S -m "feat(core): add DispatchKey and BackendComponent

Variant order is load-bearing: it encodes dispatch priority, exactly as
c10/core/DispatchKey.h does upstream."
```

---

## Task 8: rusticide-core -- DispatchKeySet

The u64 algebra from the spec: orthogonal backend and functionality
halves, resolved with two leading-zero counts.

**Files:**

- Create: `crates/rusticide-core/src/key/key_set.rs`
- Modify: `crates/rusticide-core/src/key/mod.rs`

**Interfaces:**

- Consumes: `BackendComponent`, `DispatchKey`, `NUM_BACKENDS` (Task 7).
- Produces: `DispatchKeySet` with `empty()`, `with_backend()`,
  `with_functionality()`, `union()`, `has_backend()`,
  `highest_functionality()`, `highest_backend()`, `table_index()`.

**Steps:**

- [ ] **Step 1: Write the failing test**

```rust
// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
mod tests {
    use super::DispatchKeySet;
    use crate::{BackendComponent, DispatchKey};

    #[test]
    fn an_empty_set_has_no_keys() {
        let s = DispatchKeySet::empty();
        assert!(s.highest_functionality().is_none());
        assert!(s.highest_backend().is_none());
    }

    #[test]
    fn a_set_reports_the_backend_it_was_given() {
        let s = DispatchKeySet::empty().with_backend(BackendComponent::Cpu);
        assert!(s.has_backend(BackendComponent::Cpu));
        assert!(!s.has_backend(BackendComponent::Gpu));
        assert_eq!(s.highest_backend(), Some(BackendComponent::Cpu));
    }

    #[test]
    fn the_highest_functionality_wins_regardless_of_insertion_order() {
        let a = DispatchKeySet::empty()
            .with_functionality(DispatchKey::Dense)
            .with_functionality(DispatchKey::Autograd);
        let b = DispatchKeySet::empty()
            .with_functionality(DispatchKey::Autograd)
            .with_functionality(DispatchKey::Dense);
        assert_eq!(a.highest_functionality(), Some(DispatchKey::Autograd));
        assert_eq!(b.highest_functionality(), Some(DispatchKey::Autograd));
    }

    #[test]
    fn union_is_how_multiple_dispatch_combines_operands() {
        let cpu_dense = DispatchKeySet::empty()
            .with_backend(BackendComponent::Cpu)
            .with_functionality(DispatchKey::Dense);
        let autograd = DispatchKeySet::empty()
            .with_functionality(DispatchKey::Autograd);
        let merged = cpu_dense.union(autograd);
        assert_eq!(merged.highest_functionality(), Some(DispatchKey::Autograd));
        assert_eq!(merged.highest_backend(), Some(BackendComponent::Cpu));
    }

    #[test]
    fn per_backend_functionalities_get_one_slot_per_backend() {
        let cpu = DispatchKeySet::empty()
            .with_backend(BackendComponent::Cpu)
            .with_functionality(DispatchKey::Dense);
        let gpu = DispatchKeySet::empty()
            .with_backend(BackendComponent::Gpu)
            .with_functionality(DispatchKey::Dense);
        assert_ne!(
            cpu.table_index(),
            gpu.table_index(),
            "dense-cpu and dense-gpu must not share a kernel slot"
        );
    }

    #[test]
    fn a_non_per_backend_functionality_uses_one_slot() {
        let cpu = DispatchKeySet::empty()
            .with_backend(BackendComponent::Cpu)
            .with_functionality(DispatchKey::Functionalize);
        let gpu = DispatchKeySet::empty()
            .with_backend(BackendComponent::Gpu)
            .with_functionality(DispatchKey::Functionalize);
        assert_eq!(cpu.table_index(), gpu.table_index());
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p rusticide-core key_set`
Expected: FAIL, `cannot find type 'DispatchKeySet'`.

- [ ] **Step 3: Implement `DispatchKeySet`**

```rust
use crate::{BackendComponent, DispatchKey, NUM_BACKENDS};

/// Number of functionality variants. Update alongside `DispatchKey`.
const NUM_FUNCTIONALITIES: u8 = 3;

/// Backends per per-backend functionality, plus one slot for the
/// non-per-backend case.
const BACKENDS_PER_FUNCTIONALITY: usize = 2;

/// A set of dispatch keys packed into one `u64`.
///
/// The low `NUM_BACKENDS` bits hold backends; the remaining bits hold
/// functionalities. Storing them orthogonally avoids giving every
/// (backend x functionality) pair its own bit. The runtime key is
/// reconstituted by taking the highest set bit of each half.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DispatchKeySet(u64);

impl DispatchKeySet {
    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }

    #[must_use]
    pub const fn with_backend(self, b: BackendComponent) -> Self {
        Self(self.0 | (1u64 << (b as u8 - 1)))
    }

    #[must_use]
    pub const fn with_functionality(self, k: DispatchKey) -> Self {
        Self(self.0 | (1u64 << (NUM_BACKENDS + k as u8)))
    }

    /// Multiple dispatch: OR the keysets of every operand.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    #[must_use]
    pub const fn has_backend(self, b: BackendComponent) -> bool {
        self.0 & (1u64 << (b as u8 - 1)) != 0
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The highest-priority functionality in the set.
    #[must_use]
    pub fn highest_functionality(self) -> Option<DispatchKey> {
        let bits = self.0 >> NUM_BACKENDS;
        if bits == 0 {
            return None;
        }
        let idx = 63 - bits.leading_zeros() as u8;
        match idx {
            0 => Some(DispatchKey::Dense),
            1 => Some(DispatchKey::Functionalize),
            2 => Some(DispatchKey::Autograd),
            _ => None,
        }
    }

    /// The highest-priority backend in the set.
    #[must_use]
    pub fn highest_backend(self) -> Option<BackendComponent> {
        let mask = (1u64 << NUM_BACKENDS) - 1;
        let bits = self.0 & mask;
        if bits == 0 {
            return None;
        }
        match 64 - bits.leading_zeros() {
            1 => Some(BackendComponent::Cpu),
            2 => Some(BackendComponent::Gpu),
            _ => None,
        }
    }

    /// The slot in an operator's kernel table for this keyset.
    ///
    /// Per-backend functionalities occupy `BACKENDS_PER_FUNCTIONALITY`
    /// consecutive slots; others occupy one.
    #[must_use]
    pub fn table_index(self) -> usize {
        let Some(f) = self.highest_functionality() else {
            return 0;
        };
        let mut offset = 0usize;
        for raw in 0..(f as u8) {
            offset += match raw {
                0 | 2 => BACKENDS_PER_FUNCTIONALITY,
                _ => 1,
            };
        }
        if f.is_per_backend() {
            let backend_idx =
                self.highest_backend().map_or(0, |b| b as usize - 1);
            offset + backend_idx
        } else {
            offset
        }
    }

    /// Total slots an operator table must hold.
    #[must_use]
    pub const fn table_size() -> usize {
        // Dense and Autograd are per-backend; Functionalize is not.
        BACKENDS_PER_FUNCTIONALITY * 2 + 1
    }

    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

const _: () = assert!(NUM_FUNCTIONALITIES as u32 + NUM_BACKENDS as u32 <= 64);
```

- [ ] **Step 4: Run the tests and commit**

Run: `cargo test -p rusticide-core` -- expected: 28 passed.

```bash
mise run clippy
git add crates/rusticide-core
git commit -S -m "feat(core): add DispatchKeySet

One u64 with orthogonal backend and functionality halves. Kernel
selection is two leading-zero counts and an array index, matching
c10/core/DispatchKeySet.h."
```

---

## Task 9: rusticide-core -- Tensor

**Files:**

- Create: `crates/rusticide-core/src/tensor/mod.rs`
- Create: `crates/rusticide-core/src/tensor/version.rs`
- Create: `crates/rusticide-core/src/tensor/impl_.rs`
- Create: `crates/rusticide-core/src/tensor/handle.rs`
- Modify: `crates/rusticide-core/src/lib.rs`

**Interfaces:**

- Consumes: `Storage` (6), `Shape`/`Strides` (4), `DType` (2),
  `Device` (3), `DispatchKeySet` (8).
- Produces: `Tensor` with `zeros(&[usize], DType, Device) -> Result<Tensor>`,
  `shape()`, `strides()`, `dtype()`, `device()`, `numel()`,
  `key_set()`, `storage()`, `version()`, `bump_version()`,
  `shares_storage_with()`; and `VersionCounter`.

**Steps:**

- [ ] **Step 1: Write the failing test**

`crates/rusticide-core/src/tensor/handle.rs`:

```rust
// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
mod tests {
    use super::Tensor;
    use crate::{BackendComponent, DType, Device, DispatchKey};

    #[test]
    fn zeros_has_the_requested_shape_and_dtype() {
        let t = Tensor::zeros(&[2, 3], DType::F32, Device::cpu())
            .expect("allocation succeeds");
        assert_eq!(t.shape().dims(), &[2, 3]);
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.strides().as_slice(), &[3, 1]);
    }

    #[test]
    fn a_cpu_dense_tensor_carries_the_matching_keys() {
        let t = Tensor::zeros(&[1], DType::F32, Device::cpu())
            .expect("allocation succeeds");
        let ks = t.key_set();
        assert!(ks.has_backend(BackendComponent::Cpu));
        assert_eq!(ks.highest_functionality(), Some(DispatchKey::Dense));
    }

    #[test]
    fn cloning_a_tensor_shares_storage() {
        let a = Tensor::zeros(&[4], DType::U8, Device::cpu())
            .expect("allocation succeeds");
        let b = a.clone();
        assert!(a.shares_storage_with(&b));
    }

    #[test]
    fn version_starts_at_zero_and_bumps_on_mutation() {
        let t = Tensor::zeros(&[1], DType::F32, Device::cpu())
            .expect("allocation succeeds");
        assert_eq!(t.version(), 0);
        t.bump_version();
        assert_eq!(t.version(), 1);
    }

    #[test]
    fn shared_tensors_share_one_version_counter() {
        let a = Tensor::zeros(&[1], DType::F32, Device::cpu())
            .expect("allocation succeeds");
        let b = a.clone();
        a.bump_version();
        assert_eq!(b.version(), 1, "autograd relies on this");
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p rusticide-core tensor`
Expected: FAIL, `cannot find type 'Tensor'`.

- [ ] **Step 3: Implement `VersionCounter`**

`version.rs`:

```rust
use alloc::sync::Arc;
use core::sync::atomic::{AtomicU32, Ordering};

/// Counts in-place mutations of a tensor's data.
///
/// Autograd compares a saved tensor's recorded version against the live
/// one to detect "modified by an inplace operation". Views and clones
/// share a counter, so a mutation through any of them is observed by
/// all. This is a runtime borrow check: it turns what would be a silent
/// wrong gradient into a reported error.
#[derive(Debug, Clone, Default)]
pub struct VersionCounter(Arc<AtomicU32>);

impl VersionCounter {
    #[must_use]
    pub fn new() -> Self {
        Self(Arc::new(AtomicU32::new(0)))
    }

    #[must_use]
    pub fn get(&self) -> u32 {
        self.0.load(Ordering::Acquire)
    }

    pub fn bump(&self) {
        self.0.fetch_add(1, Ordering::AcqRel);
    }
}
```

- [ ] **Step 4: Implement `TensorImpl` and `Tensor`**

`impl_.rs` holds the fields; `handle.rs` holds the public handle:

```rust
use alloc::sync::Arc;

use crate::{
    BackendComponent, DType, Device, DeviceKind, DispatchKey,
    DispatchKeySet, Result, Shape, Storage, Strides,
};

use super::version::VersionCounter;

/// The data behind a `Tensor` handle.
#[derive(Debug)]
pub(crate) struct TensorImpl {
    pub(crate) storage: Storage,
    pub(crate) shape: Shape,
    pub(crate) strides: Strides,
    pub(crate) storage_offset: usize,
    pub(crate) version: VersionCounter,
    pub(crate) key_set: DispatchKeySet,
}

/// A tensor.
///
/// Cloning is cheap and aliasing: clones share storage and a version
/// counter.
#[derive(Debug, Clone)]
pub struct Tensor(Arc<TensorImpl>);

impl Tensor {
    pub fn zeros(dims: &[usize], dtype: DType, device: Device) -> Result<Self> {
        let shape = Shape::new(dims);
        let strides = Strides::contiguous_for(&shape);
        let storage = Storage::new(dtype, device, shape.numel())?;
        let backend = match device.kind() {
            DeviceKind::Cpu => BackendComponent::Cpu,
            DeviceKind::Gpu => BackendComponent::Gpu,
        };
        let key_set = DispatchKeySet::empty()
            .with_backend(backend)
            .with_functionality(DispatchKey::Dense);
        Ok(Self(Arc::new(TensorImpl {
            storage,
            shape,
            strides,
            storage_offset: 0,
            version: VersionCounter::new(),
            key_set,
        })))
    }

    #[must_use]
    pub fn shape(&self) -> &Shape {
        &self.0.shape
    }

    #[must_use]
    pub fn strides(&self) -> &Strides {
        &self.0.strides
    }

    #[must_use]
    pub fn dtype(&self) -> DType {
        self.0.storage.dtype()
    }

    #[must_use]
    pub fn device(&self) -> Device {
        self.0.storage.device()
    }

    #[must_use]
    pub fn numel(&self) -> usize {
        self.0.shape.numel()
    }

    #[must_use]
    pub fn key_set(&self) -> DispatchKeySet {
        self.0.key_set
    }

    #[must_use]
    pub fn storage(&self) -> &Storage {
        &self.0.storage
    }

    #[must_use]
    pub fn version(&self) -> u32 {
        self.0.version.get()
    }

    pub fn bump_version(&self) {
        self.0.version.bump();
    }

    #[must_use]
    pub fn shares_storage_with(&self, other: &Self) -> bool {
        self.0.storage.same_buffer_as(&other.0.storage)
    }
}
```

- [ ] **Step 5: Run the tests and commit**

Run: `cargo test -p rusticide-core` -- expected: 33 passed.

```bash
mise run clippy
git add crates/rusticide-core
git commit -S -m "feat(core): add Tensor, TensorImpl and VersionCounter

The version counter lives on the tensor rather than on autograd
metadata so it is available even when a tensor does not require grad,
and is shared by clones so any mutation is observed by all holders."
```

---

## Task 10: rusticide-schema -- Type

`rusticide-schema` has no dependency on the core and could be built in
parallel with Tasks 2-9.

**Files:**

- Create: `crates/rusticide-schema/Cargo.toml`
- Create: `crates/rusticide-schema/README.md`
- Create: `crates/rusticide-schema/src/lib.rs`
- Create: `crates/rusticide-schema/src/ty.rs`
- Modify: `Cargo.toml`

**Interfaces:**

- Consumes: nothing.
- Produces: `BaseTy`, `Type`, `Type::parse(&str) -> Result<Type>`, and
  `Display for Type` such that `parse(s).to_string() == s`.

**Steps:**

- [ ] **Step 1: Write the failing test**

```rust
// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
mod tests {
    use super::{BaseTy, Type};

    fn round_trip(s: &str) {
        let t = Type::parse(s).expect("parses");
        assert_eq!(t.to_string(), s, "round-trip must be byte exact");
    }

    #[test]
    fn base_types_parse() {
        assert_eq!(Type::parse("Tensor"), Ok(Type::Base(BaseTy::Tensor)));
        assert_eq!(Type::parse("int"), Ok(Type::Base(BaseTy::Int)));
        assert_eq!(Type::parse("SymInt"), Ok(Type::Base(BaseTy::SymInt)));
    }

    #[test]
    fn optional_and_list_nest() {
        round_trip("Tensor?");
        round_trip("int[]");
        round_trip("int[2]");
        round_trip("Tensor?[]");
        round_trip("int[]?");
    }

    #[test]
    fn unknown_base_types_are_rejected() {
        assert!(Type::parse("Frobnicator").is_err());
    }

    #[test]
    fn every_shape_round_trips() {
        for s in ["Tensor", "float", "bool[3]", "ScalarType?", "str"] {
            round_trip(s);
        }
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p rusticide-schema`
Expected: FAIL, unresolved `Type`.

- [ ] **Step 3: Implement `BaseTy` and `Type`**

```rust
use core::fmt;

/// An atomic type in the operator schema language.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BaseTy {
    Bool,
    Device,
    DeviceIndex,
    DimVector,
    Float,
    Generator,
    GraphModule,
    Int,
    Layout,
    MemoryFormat,
    QScheme,
    Scalar,
    ScalarType,
    Storage,
    Str,
    Stream,
    SymBool,
    SymInt,
    Tensor,
}

impl BaseTy {
    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "bool" => Self::Bool,
            "Device" => Self::Device,
            "DeviceIndex" => Self::DeviceIndex,
            "DimVector" => Self::DimVector,
            "float" => Self::Float,
            "Generator" => Self::Generator,
            "GraphModule" => Self::GraphModule,
            "int" => Self::Int,
            "Layout" => Self::Layout,
            "MemoryFormat" => Self::MemoryFormat,
            "QScheme" => Self::QScheme,
            "Scalar" => Self::Scalar,
            "ScalarType" => Self::ScalarType,
            "Storage" => Self::Storage,
            "str" => Self::Str,
            "Stream" => Self::Stream,
            "SymBool" => Self::SymBool,
            "SymInt" => Self::SymInt,
            "Tensor" => Self::Tensor,
            _ => return None,
        })
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::Device => "Device",
            Self::DeviceIndex => "DeviceIndex",
            Self::DimVector => "DimVector",
            Self::Float => "float",
            Self::Generator => "Generator",
            Self::GraphModule => "GraphModule",
            Self::Int => "int",
            Self::Layout => "Layout",
            Self::MemoryFormat => "MemoryFormat",
            Self::QScheme => "QScheme",
            Self::Scalar => "Scalar",
            Self::ScalarType => "ScalarType",
            Self::Storage => "Storage",
            Self::Str => "str",
            Self::Stream => "Stream",
            Self::SymBool => "SymBool",
            Self::SymInt => "SymInt",
            Self::Tensor => "Tensor",
        }
    }
}

/// A schema type: a base type, optionally wrapped in `?` and `[]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    Base(BaseTy),
    Optional(Box<Type>),
    List { elem: Box<Type>, size: Option<usize> },
}

/// Errors from parsing schema fragments.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ParseError {
    #[error("unrecognized type: {0}")]
    UnknownType(String),
    #[error("malformed list size in: {0}")]
    BadListSize(String),
    #[error("unrecognized alias annotation: {0}")]
    BadAnnotation(String),
    #[error("malformed schema: {0}")]
    BadSchema(String),
    #[error("round-trip mismatch: parsed {parsed}, expected {expected}")]
    RoundTrip { parsed: String, expected: String },
}

impl Type {
    /// Parse a type, asserting it round-trips.
    ///
    /// The assertion is the point: a type that cannot be printed back
    /// exactly is a type whose representation lost information.
    pub fn parse(s: &str) -> Result<Self, ParseError> {
        let t = Self::parse_inner(s)?;
        let printed = t.to_string();
        if printed == s {
            Ok(t)
        } else {
            Err(ParseError::RoundTrip {
                parsed: printed,
                expected: s.to_owned(),
            })
        }
    }

    fn parse_inner(s: &str) -> Result<Self, ParseError> {
        if let Some(inner) = s.strip_suffix('?') {
            return Ok(Self::Optional(Box::new(Self::parse_inner(inner)?)));
        }
        if let Some(open) = s.rfind('[') {
            if let Some(rest) = s.strip_suffix(']') {
                let size_str = &rest[open + 1..];
                let size = if size_str.is_empty() {
                    None
                } else {
                    Some(
                        size_str
                            .parse::<usize>()
                            .map_err(|_| ParseError::BadListSize(s.to_owned()))?,
                    )
                };
                let elem = Self::parse_inner(&s[..open])?;
                return Ok(Self::List { elem: Box::new(elem), size });
            }
        }
        BaseTy::parse(s)
            .map(Self::Base)
            .ok_or_else(|| ParseError::UnknownType(s.to_owned()))
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Base(b) => write!(f, "{}", b.as_str()),
            Self::Optional(inner) => write!(f, "{inner}?"),
            Self::List { elem, size: None } => write!(f, "{elem}[]"),
            Self::List { elem, size: Some(n) } => write!(f, "{elem}[{n}]"),
        }
    }
}
```

- [ ] **Step 4: Run the tests and commit**

Add the crate to `members`, create `lib.rs` re-exporting
`{BaseTy, ParseError, Type}`.

Run: `cargo test -p rusticide-schema` -- expected: 4 passed.

```bash
mise run clippy
git add crates/rusticide-schema Cargo.toml
git commit -S -m "feat(schema): add Type with round-trip-asserted parsing

Type::parse re-prints and compares. A representation that cannot round
-trip has lost information, which is exactly the bug the assertion
catches."
```

---

## Task 11: rusticide-schema -- Annotation

This is where the borrow information lives: `Tensor(a!)` means alias set
`a`, written.

**Files:**

- Create: `crates/rusticide-schema/src/annotation.rs`
- Modify: `crates/rusticide-schema/src/lib.rs`

**Interfaces:**

- Consumes: `ParseError` (Task 10).
- Produces: `Annotation` with `alias_set: Vec<char>`, `is_write: bool`,
  `alias_set_after: Vec<char>`, `parse(&str)`, and `Display`.

**Steps:**

- [ ] **Step 1: Write the failing test**

```rust
// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
mod tests {
    use super::Annotation;

    fn round_trip(s: &str) {
        let a = Annotation::parse(s).expect("parses");
        assert_eq!(a.to_string(), s);
    }

    #[test]
    fn a_read_only_alias_has_no_write_flag() {
        let a = Annotation::parse("a").expect("parses");
        assert_eq!(a.alias_set, vec!['a']);
        assert!(!a.is_write);
        round_trip("a");
    }

    #[test]
    fn a_bang_marks_a_write() {
        let a = Annotation::parse("a!").expect("parses");
        assert!(a.is_write);
        round_trip("a!");
    }

    #[test]
    fn a_mutable_reference_may_belong_to_only_one_alias_set() {
        assert!(
            Annotation::parse("a|b!").is_err(),
            "aliasing XOR mutation: this is Rust's exclusivity rule"
        );
    }

    #[test]
    fn a_read_only_annotation_may_name_several_sets() {
        let a = Annotation::parse("a|b").expect("parses");
        assert_eq!(a.alias_set, vec!['a', 'b']);
        round_trip("a|b");
    }

    #[test]
    fn the_wildcard_escape_round_trips() {
        round_trip("a -> *");
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p rusticide-schema annotation`
Expected: FAIL, `cannot find type 'Annotation'`.

- [ ] **Step 3: Implement `Annotation`**

```rust
use core::fmt;

use crate::ParseError;

/// An alias annotation on a schema argument or return.
///
/// `Tensor(a!)` parses to alias set `['a']` with `is_write` set. The
/// invariant enforced in `parse` -- a write may name only one alias set
/// -- is aliasing XOR mutation, the same rule Rust's borrow checker
/// enforces statically.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Annotation {
    pub alias_set: Vec<char>,
    pub is_write: bool,
    pub alias_set_after: Vec<char>,
}

impl Annotation {
    pub fn parse(s: &str) -> Result<Self, ParseError> {
        let bad = || ParseError::BadAnnotation(s.to_owned());
        let (before, after) = match s.split_once(" -> ") {
            Some((b, a)) => (b, Some(a)),
            None => (s, None),
        };
        let (set_str, is_write) = match before.strip_suffix('!') {
            Some(rest) => (rest, true),
            None => (before, false),
        };
        let alias_set = parse_set(set_str).ok_or_else(bad)?;
        if is_write && alias_set.len() > 1 {
            return Err(bad());
        }
        let alias_set_after = match after {
            None => Vec::new(),
            Some("*") => vec!['*'],
            Some(a) => parse_set(a).ok_or_else(bad)?,
        };
        if alias_set.len() > 1 && alias_set_after.len() > 1 {
            return Err(bad());
        }
        Ok(Self { alias_set, is_write, alias_set_after })
    }
}

fn parse_set(s: &str) -> Option<Vec<char>> {
    if s.is_empty() {
        return None;
    }
    s.split('|')
        .map(|part| {
            let mut it = part.chars();
            match (it.next(), it.next()) {
                (Some(c), None) if c.is_ascii_lowercase() => Some(c),
                _ => None,
            }
        })
        .collect()
}

impl fmt::Display for Annotation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let set: Vec<String> =
            self.alias_set.iter().map(ToString::to_string).collect();
        write!(f, "{}", set.join("|"))?;
        if self.is_write {
            write!(f, "!")?;
        }
        if !self.alias_set_after.is_empty() {
            let after: Vec<String> = self
                .alias_set_after
                .iter()
                .map(ToString::to_string)
                .collect();
            write!(f, " -> {}", after.join("|"))?;
        }
        Ok(())
    }
}
```

- [ ] **Step 4: Run the tests and commit**

Run: `cargo test -p rusticide-schema` -- expected: 9 passed.

```bash
mise run clippy
git add crates/rusticide-schema
git commit -S -m "feat(schema): add Annotation with the exclusivity rule

A write may name only one alias set. That constraint is aliasing XOR
mutation, declared in the schema DSL and enforced at parse time across
the whole operator surface."
```

---

## Task 12: rusticide-schema -- Argument, Return, OperatorName

**Files:**

- Create: `crates/rusticide-schema/src/argument.rs`
- Create: `crates/rusticide-schema/src/name.rs`
- Modify: `crates/rusticide-schema/src/lib.rs`

**Interfaces:**

- Consumes: `Type`, `Annotation`, `ParseError`.
- Produces: `Argument { name, ty, annotation, default }`,
  `Return { name, ty, annotation }`, `Arguments { positional, kwarg_only,
  out }`, `OperatorName { name, overload }`, each with `parse` and
  `Display`.

**Steps:**

- [ ] **Step 1: Write the failing tests**

```rust
// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
mod tests {
    use super::{Argument, Arguments};

    fn round_trip(s: &str) {
        let a = Argument::parse(s).expect("parses");
        assert_eq!(a.to_string(), s);
    }

    #[test]
    fn a_plain_argument_has_a_type_and_a_name() {
        let a = Argument::parse("Tensor self").expect("parses");
        assert_eq!(a.name, "self");
        assert!(a.annotation.is_none());
        assert!(a.default.is_none());
        round_trip("Tensor self");
    }

    #[test]
    fn an_annotated_mutable_argument_carries_its_alias_set() {
        let a = Argument::parse("Tensor(a!) self").expect("parses");
        let ann = a.annotation.as_ref().expect("has annotation");
        assert!(ann.is_write);
        round_trip("Tensor(a!) self");
    }

    #[test]
    fn defaults_round_trip_verbatim() {
        round_trip("bool non_blocking=False");
        round_trip("int dim=0");
        round_trip("int[1] dim=[0]");
        round_trip("ScalarType? dtype=None");
    }

    #[test]
    fn kwarg_only_arguments_follow_the_star() {
        let args =
            Arguments::parse("Tensor self, *, int dim=0").expect("parses");
        assert_eq!(args.positional.len(), 1);
        assert_eq!(args.kwarg_only.len(), 1);
        assert_eq!(args.to_string(), "Tensor self, *, int dim=0");
    }

    #[test]
    fn out_arguments_are_kwarg_only_and_annotated() {
        let args = Arguments::parse("Tensor self, *, Tensor(a!) out")
            .expect("parses");
        assert_eq!(args.out.len(), 1);
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p rusticide-schema argument`
Expected: FAIL, unresolved `Argument`.

- [ ] **Step 3: Implement `Argument` and `Arguments`**

An argument is `Type[(annotation)] name[=default]`. Split the annotation
off the type by finding the parenthesised suffix on the type token.

```rust
use core::fmt;

use crate::{Annotation, ParseError, Type};

/// One argument in an operator schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Argument {
    pub name: String,
    pub ty: Type,
    pub annotation: Option<Annotation>,
    pub default: Option<String>,
}

impl Argument {
    pub fn parse(s: &str) -> Result<Self, ParseError> {
        let bad = || ParseError::BadSchema(s.to_owned());
        let (type_and_name, default) = match s.split_once('=') {
            Some((lhs, rhs)) => (lhs, Some(rhs.to_owned())),
            None => (s, None),
        };
        let (type_str, name) =
            type_and_name.rsplit_once(' ').ok_or_else(bad)?;
        let (ty, annotation) = split_annotation(type_str)?;
        Ok(Self {
            name: name.to_owned(),
            ty,
            annotation,
            default,
        })
    }

    #[must_use]
    pub fn is_write(&self) -> bool {
        self.annotation.as_ref().is_some_and(|a| a.is_write)
    }
}

/// Split `Tensor(a!)` into its type and its annotation.
fn split_annotation(s: &str) -> Result<(Type, Option<Annotation>), ParseError> {
    let bad = || ParseError::BadSchema(s.to_owned());
    let Some(open) = s.find('(') else {
        return Ok((Type::parse(s)?, None));
    };
    let close = s.rfind(')').ok_or_else(bad)?;
    let ann = Annotation::parse(&s[open + 1..close])?;
    let mut ty_str = String::with_capacity(s.len());
    ty_str.push_str(&s[..open]);
    ty_str.push_str(&s[close + 1..]);
    Ok((Type::parse(&ty_str)?, Some(ann)))
}

impl fmt::Display for Argument {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The annotation attaches to the base type, before any `?`/`[]`
        // suffix: `Tensor(a!)[]`, never `Tensor[](a!)`.
        let printed = self.ty.to_string();
        match &self.annotation {
            None => write!(f, "{printed}")?,
            Some(a) => {
                let base_len = printed
                    .find(['?', '['])
                    .unwrap_or(printed.len());
                write!(
                    f,
                    "{}({}){}",
                    &printed[..base_len],
                    a,
                    &printed[base_len..]
                )?;
            }
        }
        write!(f, " {}", self.name)?;
        if let Some(d) = &self.default {
            write!(f, "={d}")?;
        }
        Ok(())
    }
}

/// The argument list of a schema, split at the `*` marker.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Arguments {
    pub positional: Vec<Argument>,
    pub kwarg_only: Vec<Argument>,
    pub out: Vec<Argument>,
}

impl Arguments {
    pub fn parse(s: &str) -> Result<Self, ParseError> {
        let mut out = Self::default();
        if s.trim().is_empty() {
            return Ok(out);
        }
        let mut after_star = false;
        for part in split_top_level(s) {
            let part = part.trim();
            if part == "*" {
                after_star = true;
                continue;
            }
            let arg = Argument::parse(part)?;
            if !after_star {
                out.positional.push(arg);
            } else if arg.is_write() {
                out.out.push(arg);
            } else {
                out.kwarg_only.push(arg);
            }
        }
        Ok(out)
    }
}

/// Split on commas that are not inside brackets, so `int[1] x=[0, 1]`
/// stays one argument.
fn split_top_level(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    for (i, c) in s.char_indices() {
        match c {
            '[' | '(' => depth += 1,
            ']' | ')' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&s[start..]);
    parts
}

impl fmt::Display for Arguments {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts: Vec<String> =
            self.positional.iter().map(ToString::to_string).collect();
        if !self.kwarg_only.is_empty() || !self.out.is_empty() {
            parts.push("*".to_owned());
            parts.extend(self.kwarg_only.iter().map(ToString::to_string));
            parts.extend(self.out.iter().map(ToString::to_string));
        }
        write!(f, "{}", parts.join(", "))
    }
}
```

- [ ] **Step 4: Implement `OperatorName` and `Return`**

`name.rs`:

```rust
use core::fmt;

/// An operator's name and optional overload tag: `add.Tensor`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OperatorName {
    pub name: String,
    pub overload: Option<String>,
}

impl OperatorName {
    #[must_use]
    pub fn parse(s: &str) -> Self {
        match s.split_once('.') {
            Some((n, o)) => Self {
                name: n.to_owned(),
                overload: Some(o.to_owned()),
            },
            None => Self { name: s.to_owned(), overload: None },
        }
    }

    /// Trailing `_` marks an in-place operator.
    #[must_use]
    pub fn is_inplace(&self) -> bool {
        self.name.ends_with('_') && !self.name.ends_with("__")
    }
}

impl fmt::Display for OperatorName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.overload {
            Some(o) => write!(f, "{}.{}", self.name, o),
            None => write!(f, "{}", self.name),
        }
    }
}
```

`Return` mirrors `Argument` but its name is optional; put it in
`argument.rs` beside `Argument` since they change together.

- [ ] **Step 5: Run the tests and commit**

Run: `cargo test -p rusticide-schema` -- expected: 14 passed.

```bash
mise run clippy
git add crates/rusticide-schema
git commit -S -m "feat(schema): add Argument, Arguments, Return and OperatorName

Commas inside brackets do not separate arguments, so int[1] dim=[0, 1]
stays one argument; splitting is bracket-depth aware."
```

---

## Task 13: rusticide-schema -- FunctionSchema and the corpus round-trip

The milestone with the unambiguous pass condition.

**Files:**

- Create: `crates/rusticide-schema/src/schema.rs`
- Create: `crates/rusticide-schema/tests/corpus.rs`
- Create: `crates/rusticide-schema/tests/data/native_functions.yaml`
- Modify: `crates/rusticide-schema/src/lib.rs`

**Interfaces:**

- Consumes: `Arguments`, `Return`, `OperatorName`, `ParseError`.
- Produces: `FunctionSchema { name, arguments, returns }` with
  `parse(&str)`, `Display`, and `is_mutable()`.

**Steps:**

- [ ] **Step 1: Vendor the corpus**

```bash
mkdir -p crates/rusticide-schema/tests/data
grep '^- func:' \
  /Volumes/Development/github.com/pytorch/pytorch/aten/src/ATen/native/native_functions.yaml \
  | sed 's/^- func: //' \
  > crates/rusticide-schema/tests/data/schemas.txt
wc -l crates/rusticide-schema/tests/data/schemas.txt
```

Expected: 2585 lines. Storing the extracted schema lines rather than the
631 KB YAML keeps the fixture to what the test actually reads.

- [ ] **Step 2: Write the failing corpus test**

`crates/rusticide-schema/tests/corpus.rs`:

```rust
// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Round-trip every operator schema PyTorch declares.
//!
//! This is the acceptance test for the parser: parse, re-print, require
//! byte equality. A schema that does not round-trip is a schema whose
//! representation lost information.

use rusticide_schema::FunctionSchema;

const CORPUS: &str = include_str!("data/schemas.txt");

#[test]
fn every_schema_round_trips_byte_exactly() {
    let mut failures = Vec::new();
    let mut count = 0usize;
    for line in CORPUS.lines().map(str::trim).filter(|l| !l.is_empty()) {
        count += 1;
        match FunctionSchema::parse(line) {
            Ok(s) => {
                let printed = s.to_string();
                if printed != line {
                    failures.push(format!("  got: {printed}\n  want: {line}"));
                }
            }
            Err(e) => failures.push(format!("  {line}\n  error: {e}")),
        }
    }
    assert!(count >= 2585, "corpus shrank: only {count} schemas");
    assert!(
        failures.is_empty(),
        "{} of {count} schemas failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn the_corpus_contains_the_annotation_shapes_we_rely_on() {
    let mutable = CORPUS.lines().filter(|l| l.contains("(a!)")).count();
    assert!(mutable > 900, "expected ~981 mutable schemas, saw {mutable}");
}
```

- [ ] **Step 3: Run to verify it fails**

Run: `cargo test -p rusticide-schema --test corpus`
Expected: FAIL, `cannot find type 'FunctionSchema'`.

- [ ] **Step 4: Implement `FunctionSchema`**

```rust
use core::fmt;

use crate::{Arguments, OperatorName, ParseError, Return};

/// A complete operator signature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionSchema {
    pub name: OperatorName,
    pub arguments: Arguments,
    pub returns: Vec<Return>,
}

impl FunctionSchema {
    pub fn parse(s: &str) -> Result<Self, ParseError> {
        let bad = || ParseError::BadSchema(s.to_owned());
        let open = s.find('(').ok_or_else(bad)?;
        let arrow = s.rfind(") -> ").ok_or_else(bad)?;
        let name = OperatorName::parse(&s[..open]);
        let arguments = Arguments::parse(&s[open + 1..arrow])?;
        let returns = parse_returns(&s[arrow + 5..])?;
        let schema = Self { name, arguments, returns };
        let printed = schema.to_string();
        if printed == s {
            Ok(schema)
        } else {
            Err(ParseError::RoundTrip {
                parsed: printed,
                expected: s.to_owned(),
            })
        }
    }

    /// Whether any argument is annotated as written.
    #[must_use]
    pub fn is_mutable(&self) -> bool {
        self.arguments
            .positional
            .iter()
            .chain(&self.arguments.kwarg_only)
            .chain(&self.arguments.out)
            .any(Argument::is_write)
    }
}
```

`parse_returns` handles the three forms the corpus uses: `()`, a single
type, and a parenthesised tuple with optional names. `Display` prints
`{name}({arguments}) -> {returns}`, reproducing the same three forms.

- [ ] **Step 5: Run the corpus test until it passes**

Run: `cargo test -p rusticide-schema --test corpus -- --nocapture`

Expect several iterations. The failure output prints each mismatch as
`got`/`want`, which localises the defect to one printing rule at a time.

Expected final state: `2585 schemas`, 0 failures.

- [ ] **Step 6: Commit**

```bash
mise run clippy
git add crates/rusticide-schema
git commit -S -m "feat(schema): add FunctionSchema and the corpus round-trip

All 2,585 operator schemas from PyTorch's native_functions.yaml parse
and re-print byte-exactly. This is SP1's acceptance criterion for the
parser, and it delivers the alias annotations as a side effect."
```

---

## Task 14: rusticide-dispatch -- IValue and Stack

**Files:**

- Create: `crates/rusticide-dispatch/Cargo.toml`
- Create: `crates/rusticide-dispatch/README.md`
- Create: `crates/rusticide-dispatch/src/lib.rs`
- Create: `crates/rusticide-dispatch/src/ivalue.rs`
- Create: `crates/rusticide-dispatch/src/stack.rs`
- Modify: `Cargo.toml`

**Interfaces:**

- Consumes: `Tensor`, `DType`, `Device` from `rusticide-core`.
- Produces: `IValue` enum, `Stack` with `push`, `pop_tensor`, `pop_int`,
  `len`, and `IValue::as_tensor(&self) -> Option<&Tensor>`.

**Steps:**

- [ ] **Step 1: Write the failing test**

```rust
// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
mod tests {
    use rusticide_core::{DType, Device, Tensor};

    use super::IValue;

    #[test]
    fn an_ivalue_holds_a_tensor_and_gives_it_back() {
        let t = Tensor::zeros(&[2], DType::F32, Device::cpu())
            .expect("allocation succeeds");
        let v = IValue::Tensor(t.clone());
        let got = v.as_tensor().expect("holds a tensor");
        assert!(got.shares_storage_with(&t));
    }

    #[test]
    fn asking_for_the_wrong_variant_yields_none() {
        assert!(IValue::Int(3).as_tensor().is_none());
        assert!(IValue::None.as_tensor().is_none());
    }

    #[test]
    fn scalars_round_trip_through_their_accessors() {
        assert_eq!(IValue::Int(7).as_int(), Some(7));
        assert_eq!(IValue::Double(0.5).as_double(), Some(0.5));
        assert_eq!(IValue::Bool(true).as_bool(), Some(true));
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p rusticide-dispatch`
Expected: FAIL, `cannot find type 'IValue'`.

- [ ] **Step 3: Implement `IValue`**

```rust
use rusticide_core::{DType, Device, Tensor};

/// A dynamically typed value on the dispatch stack.
///
/// PyTorch hand-rolls this as a tagged union with a manual `Tag` field
/// and hand-written payload management. In Rust it is an ordinary enum,
/// which is one of the places the port improves on the original.
#[derive(Debug, Clone)]
pub enum IValue {
    None,
    Tensor(Tensor),
    Int(i64),
    Double(f64),
    Bool(bool),
    Str(String),
    IntList(Vec<i64>),
    TensorList(Vec<Tensor>),
    Device(Device),
    DType(DType),
}

impl IValue {
    #[must_use]
    pub const fn as_tensor(&self) -> Option<&Tensor> {
        match self {
            Self::Tensor(t) => Some(t),
            _ => None,
        }
    }

    #[must_use]
    pub const fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(i) => Some(*i),
            _ => None,
        }
    }

    #[must_use]
    pub const fn as_double(&self) -> Option<f64> {
        match self {
            Self::Double(d) => Some(*d),
            _ => None,
        }
    }

    #[must_use]
    pub const fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }
}
```

- [ ] **Step 4: Implement `Stack`**

`stack.rs` wraps `Vec<IValue>` with typed pops that return
`Option`, keeping the boxed calling convention explicit. Write tests for
push/pop ordering and for a typed pop on the wrong variant returning
`None` without consuming the stack.

- [ ] **Step 5: Run the tests and commit**

Run: `cargo test -p rusticide-dispatch` -- expected: 6 passed.

```bash
mise run clippy
git add crates/rusticide-dispatch Cargo.toml
git commit -S -m "feat(dispatch): add IValue and Stack"
```

---

## Task 15: rusticide-dispatch -- KernelFunction and the aliasing guard

**Files:**

- Create: `crates/rusticide-dispatch/src/kernel.rs`
- Create: `crates/rusticide-dispatch/src/aliasing.rs`
- Modify: `crates/rusticide-dispatch/src/lib.rs`

**Interfaces:**

- Consumes: `Stack`, `IValue`, `Tensor`.
- Produces: `KernelFunction` wrapping
  `Arc<dyn Fn(&mut Stack) + Send + Sync>`, and
  `aliasing::distinct_storages(&[&Tensor]) -> bool`.

**Steps:**

- [ ] **Step 1: Write the failing test for the guard**

```rust
// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
mod tests {
    use rusticide_core::{DType, Device, Tensor};

    use super::distinct_storages;

    #[test]
    fn separate_tensors_have_distinct_storages() {
        let a = Tensor::zeros(&[1], DType::F32, Device::cpu())
            .expect("allocation succeeds");
        let b = Tensor::zeros(&[1], DType::F32, Device::cpu())
            .expect("allocation succeeds");
        assert!(distinct_storages(&[&a, &b]));
    }

    #[test]
    fn a_tensor_aliased_with_itself_is_detected() {
        let a = Tensor::zeros(&[1], DType::F32, Device::cpu())
            .expect("allocation succeeds");
        let b = a.clone();
        assert!(
            !distinct_storages(&[&a, &b]),
            "a.add_(a) must be seen, or the RwLock deadlocks"
        );
    }

    #[test]
    fn a_single_operand_is_trivially_distinct() {
        let a = Tensor::zeros(&[1], DType::F32, Device::cpu())
            .expect("allocation succeeds");
        assert!(distinct_storages(&[&a]));
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p rusticide-dispatch aliasing`
Expected: FAIL, `cannot find function 'distinct_storages'`.

- [ ] **Step 3: Implement the guard**

```rust
use rusticide_core::Tensor;

/// Whether every operand is backed by a different allocation.
///
/// `Storage` is an `Arc<RwLock<..>>`, so a call like `a.add_(a)` would
/// take a write and a read lock on one buffer and deadlock. Kernels
/// consult this before locking and degrade to a single write lock when
/// operands coincide. Operand counts are tiny, so the quadratic scan is
/// cheaper than allocating a set.
#[must_use]
pub fn distinct_storages(tensors: &[&Tensor]) -> bool {
    for (i, a) in tensors.iter().enumerate() {
        for b in &tensors[i + 1..] {
            if a.shares_storage_with(b) {
                return false;
            }
        }
    }
    true
}
```

- [ ] **Step 4: Implement `KernelFunction`**

```rust
use alloc::sync::Arc;

use crate::Stack;

/// A registered kernel, called through the boxed convention.
#[derive(Clone)]
pub struct KernelFunction(Arc<dyn Fn(&mut Stack) + Send + Sync>);

impl KernelFunction {
    #[must_use]
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&mut Stack) + Send + Sync + 'static,
    {
        Self(Arc::new(f))
    }

    pub fn call(&self, stack: &mut Stack) {
        (self.0)(stack);
    }
}

impl core::fmt::Debug for KernelFunction {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("KernelFunction(..)")
    }
}
```

- [ ] **Step 5: Run the tests and commit**

Run: `cargo test -p rusticide-dispatch` -- expected: 10 passed.

```bash
mise run clippy
git add crates/rusticide-dispatch
git commit -S -m "feat(dispatch): add KernelFunction and the self-aliasing guard

Storage is Arc<RwLock<..>>, so a.add_(a) would deadlock on a write plus
a read of one buffer. The guard belongs in the calling convention, not
in each kernel."
```

---

## Task 16: rusticide-dispatch -- Dispatcher and end-to-end add

SP1's second acceptance criterion: `add` reaches a CPU kernel through
the real table.

**Files:**

- Create: `crates/rusticide-dispatch/src/entry.rs`
- Create: `crates/rusticide-dispatch/src/dispatcher.rs`
- Create: `crates/rusticide-dispatch/tests/end_to_end.rs`
- Modify: `crates/rusticide-dispatch/src/lib.rs`

**Interfaces:**

- Consumes: `KernelFunction`, `Stack`, `IValue`, `DispatchKeySet`,
  `Tensor`, `distinct_storages`.
- Produces: `OperatorEntry::new(OperatorName)`,
  `OperatorEntry::register(DispatchKeySet, KernelFunction)`,
  `OperatorEntry::lookup(DispatchKeySet) -> Option<&KernelFunction>`,
  `Dispatcher::global() -> &'static Dispatcher`,
  `Dispatcher::register(&str, DispatchKeySet, KernelFunction)`,
  `Dispatcher::call(&str, &mut Stack) -> Result<(), DispatchError>`.

**Steps:**

- [ ] **Step 1: Write the failing end-to-end test**

```rust
// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! SP1's acceptance test: `add` dispatches through the real table.

use rusticide_core::{
    BackendComponent, DType, Device, DispatchKey, DispatchKeySet, Tensor,
};
use rusticide_dispatch::{Dispatcher, IValue, KernelFunction, Stack};

fn cpu_dense() -> DispatchKeySet {
    DispatchKeySet::empty()
        .with_backend(BackendComponent::Cpu)
        .with_functionality(DispatchKey::Dense)
}

fn register_add() {
    Dispatcher::global().register(
        "add",
        cpu_dense(),
        KernelFunction::new(|stack: &mut Stack| {
            let rhs = stack.pop_tensor().expect("rhs is a tensor");
            let lhs = stack.pop_tensor().expect("lhs is a tensor");
            let out = Tensor::zeros(
                lhs.shape().dims(),
                lhs.dtype(),
                lhs.device(),
            )
            .expect("allocation succeeds");
            {
                let a = lhs.storage().read();
                let b = rhs.storage().read();
                let mut o = out.storage().write();
                for i in 0..o.len() {
                    o[i] = a[i].wrapping_add(b[i]);
                }
            }
            stack.push(IValue::Tensor(out));
        }),
    );
}

#[test]
fn add_dispatches_to_the_cpu_kernel() {
    register_add();
    let a = Tensor::zeros(&[4], DType::U8, Device::cpu())
        .expect("allocation succeeds");
    let b = Tensor::zeros(&[4], DType::U8, Device::cpu())
        .expect("allocation succeeds");
    a.storage().write()[0] = 2;
    b.storage().write()[0] = 3;

    let mut stack = Stack::new();
    stack.push(IValue::Tensor(a));
    stack.push(IValue::Tensor(b));
    Dispatcher::global().call("add", &mut stack).expect("op is registered");

    let out = stack.pop_tensor().expect("kernel pushed a tensor");
    assert_eq!(out.storage().read()[0], 5);
}

#[test]
fn calling_an_unregistered_operator_is_an_error() {
    let mut stack = Stack::new();
    assert!(Dispatcher::global().call("nonexistent", &mut stack).is_err());
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p rusticide-dispatch --test end_to_end`
Expected: FAIL, `cannot find type 'Dispatcher'`.

- [ ] **Step 3: Implement `OperatorEntry`**

A flat table of `Option<KernelFunction>` sized by
`DispatchKeySet::table_size()`, indexed by `keyset.table_index()`.

- [ ] **Step 4: Implement `Dispatcher`**

A process-global registry behind `std::sync::OnceLock<Dispatcher>`, with
a `RwLock<HashMap<String, OperatorEntry>>` inside. `call` computes the
keyset from the tensors on the stack by union, looks up the slot, and
invokes the kernel. Registration errors and missing operators are a
`DispatchError` enum built with `thiserror`, never a panic.

- [ ] **Step 5: Run the tests and commit**

Run: `cargo test -p rusticide-dispatch` -- expected: all pass.

```bash
mise run clippy
git add crates/rusticide-dispatch
git commit -S -m "feat(dispatch): add Dispatcher and OperatorEntry

add now reaches a CPU kernel through the real dispatch table, which is
SP1's second acceptance criterion."
```

---

## Task 17: Port the eager Tensor API and add rusticide-io

> **Replaced 2026-10-22.** The original Task 17 (rename `rusticide-ggml`
> to `rusticide-io`) and Task 18 were superseded by a re-plan after the
> legacy `Tensor` API turned out to be larger than "eight `use` lines".
> Human decision: build `rusticide-io` from the deleted reader's GGUF
> layer, and port the old eager Tensor API into `rusticide-core` so the
> leftover crates compile. No rename happened: `rusticide-ggml` stays the
> legacy compute graph. This section records the outcome. Commit ranges
> are within `7aefefd..HEAD`.

Each sub-task shipped with review and the 100% coverage gate.

- **17a** `d91988f` -- `rusticide-core` element-access error variants and
  the sealed `Element` trait (`f32`, `f16`).
- **17b** `645860e..c671766` -- `Tensor::from_slice`, `get`, `set`,
  `to_vec`; fixes that saturate `Shape::numel` and contiguous strides.
- **17c** `c14858d` -- `deep_copy` and the `EagerOps` trait with
  `reshape` and `contiguous` (copying, per the design's tier-1 value
  semantics).
- **17d** `a3fd23d..56d5217` -- `EagerOps::add` (NumPy broadcasting, F32)
  and 2-D `matmul`; outputs allocated fallibly; index-order tests.
- **17e** `276aefd` -- `rusticide-quantize` rebound to core and split by
  format (Q4_0, Q4_K, Q6_K); K-quant dequantizers reject numel not a multiple of 256
  (`QuantizeError::UnalignedShape`); `quantize_q4_0` is fallible.
- **17f** `8fd4ee8` -- `rusticide-io` GGUF value layer (`GgufValue`, typed
  reads); array nesting capped (`MAX_ARRAY_DEPTH`, `NestingTooDeep`).
- **17m** `e802ede..313aadc` -- memory hardening from the rust-skills
  audit: storage guard newtypes, fallible `to_vec` and dequantizer
  allocation, ordered paired read locks.
- **17g** `3b29b04` -- `GgufFile::from_bytes`: header, metadata and tensor
  infos, with typed `GgufError`s.
- **17h** `b7b1d85` -- GGUF tensor materialization into core `Tensor`s,
  dequantizing the quantized types through `rusticide-quantize`.

## Task 18: Rewire quantize, sampling, ggml, the tokenizer and fuzz

- **18a** `88e4c69..435b70a` -- `rusticide-sampling` rebound to core and
  split by concern; `sample_with_temp` rejects non-finite logits.
- **18b** `df4e927..1132748` -- `rusticide-ggml` compute graph rebound to
  core and split; `GetRows` indices validated; `set_input` accepts input
  nodes only and `forward` recomputes every non-input node.
- **18c** `b043ce7` -- ggml math operators (unary, binary with
  broadcasting, RMSNorm, RoPE, row gather) ported onto core, allocated
  fallibly.
- **18d** `2d17dc2..a03e89f` -- `rusticide-tokenizer` reads GGUF metadata
  from `rusticide-io` and no longer depends on `Model`; non-finite
  scores rejected.
- **18e** `7e66094` -- the fuzz targets retargeted at `rusticide-io`.
- **18f** (this commit) -- docs, `AGENTS.md` and the full gate. Every
  crate on disk except `rusticide-fuzz` is now a workspace member.

### Rulings that bind later work

- The four eager operators (`add`, `matmul`, `reshape`, `contiguous`) live
  behind the `EagerOps` trait, which is the SP2 removal seam. It is not
  re-exported from the `rusticide` facade. `from_slice`, `get`, `set`,
  `to_vec`, `ndim` and `deep_copy` are inherent core data-access API and
  stay in SP2.
- Element typing is strict: only `f32` and `f16`, and `get::<f32>` on an
  F16 tensor is an error.
- `Shape::numel` and contiguous strides saturate instead of overflowing;
  `Storage::new` and `from_slice` then reject the saturated size.
- Storage guard newtypes expose only `[u8]`, so safe code cannot break
  the `len == numel * size` invariant.
- Paired read locks (eager ops and kernels alike) go through
  `Storage::with_read_pair`: global storage-address order, one guard for
  the same storage. The dispatcher-level aliasing check is deferred to SP2.
- Kernels return `Result<(), DispatchError>`; `Dispatcher::call` propagates
  it. GGUF rejects duplicate metadata keys, and the tokenizer rejects any
  token over 1024 bytes.
- GGUF: tensor offsets are relative to the aligned data section; a
  present but invalid `general.alignment` is rejected (absent means 32);
  duplicate tensor names are rejected; only version 3 is accepted;
  misaligned or out-of-range tensor offsets are rejected.
- Q4_0 uses the ggml nibble layout: element j in the low nibble of
  `qs[j]`, element j+16 in the high nibble.
- The tokenizer rejects non-finite scores read from GGUF metadata.
- The ggml graph recomputes on every `forward`, and `set_input` accepts
  input nodes only (`GraphError::NotAnInput`).
- Deferred to SP2: the allocation ceiling and the GGUF metadata
  amplification; see the design document's "Deferred from SP1".

---

## Task 19: The rusticide facade and CLI

> **Amended 2026-10-22.** `crates/rusticide` already exists: a workspace
> member with a dependency-free `--help`/`--version` CLI (`src/lib.rs`
> `run()`, unit-tested; thin `src/main.rs`), added so the release pipeline
> had a binary to ship. This task reshapes that crate instead of creating
> it: the steps below still apply, but "Create" means "rewrite", and the
> existing `run()` tests are replaced by the `clap` CLI's. One contract is
> new and binding: `rusticide --version` must print exactly
> `rusticide <version>`, because `mise run release:build` smoke-runs every
> release binary against that string. `--version` must therefore work
> with the `cli` feature, which is how release builds compile the binary.

**Files:**

- Create: `crates/rusticide/Cargo.toml`
- Create: `crates/rusticide/README.md`
- Create: `crates/rusticide/src/lib.rs`
- Create: `crates/rusticide/src/main.rs`
- Create: `crates/rusticide/src/cli/mod.rs`
- Create: `crates/rusticide/src/cli/cmd/{mod,schema,version}.rs`
- Modify: `Cargo.toml`

**Interfaces:**

- Consumes: every published crate.
- Produces: the `rusticide` library facade and the `rusticide` binary.

**Steps:**

- [ ] **Step 1: Write the manifest with a feature-gated binary**

Following the glaucus pattern: only `rusticide-core` is non-optional, and
the binary requires the `cli` feature so library consumers never pull
`clap`.

```toml
[features]
default = ["dispatch", "schema"]
dispatch = ["dep:rusticide-dispatch"]
schema = ["dep:rusticide-schema"]
io = ["dep:rusticide-io"]
cli = ["schema", "dispatch", "dep:anyhow", "dep:clap"]

[dependencies]
anyhow = { workspace = true, optional = true }
clap = { workspace = true, optional = true }
rusticide-core.workspace = true
rusticide-dispatch = { workspace = true, optional = true }
rusticide-io = { workspace = true, optional = true }
rusticide-schema = { workspace = true, optional = true }

[[bin]]
name = "rusticide"
path = "src/main.rs"
required-features = ["cli"]
```

- [ ] **Step 2: Write a failing doctest in the facade**

`src/lib.rs`:

````rust
//! # Rusticide -- a pure-Rust deep learning framework
//!
//! ```
//! use rusticide::{DType, Device, Tensor};
//!
//! let t = Tensor::zeros(&[2, 3], DType::F32, Device::cpu())?;
//! assert_eq!(t.shape().dims(), &[2, 3]);
//! # Ok::<(), rusticide::Error>(())
//! ```

pub use rusticide_core::{
    BackendComponent, DType, Device, DeviceKind, DispatchKey,
    DispatchKeySet, Error, Layout, MemoryFormat, Result, Shape, Storage,
    Strides, Tensor,
};

#[cfg(feature = "dispatch")]
pub use rusticide_dispatch as dispatch;

#[cfg(feature = "schema")]
pub use rusticide_schema as schema;
````

- [ ] **Step 3: Run the doctest to verify it fails, then passes**

Run: `cargo test -p rusticide --doc`
Expected first: FAIL, unresolved crate. After Step 2's manifest and
re-exports: PASS.

- [ ] **Step 4: Implement the CLI**

`src/cli/cmd/schema.rs` parses a schema string from argv and prints its
parsed form, dogfooding `rusticide-schema`:

```rust
// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use anyhow::Result;
use rusticide_schema::FunctionSchema;

/// Parse an operator schema and print its structure.
pub(crate) fn run(schema: &str) -> Result<()> {
    let parsed = FunctionSchema::parse(schema)?;
    println!("name:      {}", parsed.name);
    println!("mutable:   {}", parsed.is_mutable());
    println!("arguments: {}", parsed.arguments);
    println!("round-trip: {}", parsed);
    Ok(())
}
```

`src/cli/mod.rs` defines the `clap` parser with `schema` and `version`
subcommands; `src/main.rs` is a thin entry point that returns the
process exit code.

- [ ] **Step 5: Verify the binary and commit**

```bash
cargo run -p rusticide --features cli -- schema 'add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor'
```

Expected: the parsed structure, with `round-trip` echoing the input.

```bash
mise run clippy
git add crates/rusticide Cargo.toml
git commit -S -m "feat(rusticide): add the facade crate with an embedded CLI

Only rusticide-core is a non-optional dependency and the binary
requires the cli feature, so library consumers do not pull clap."
```

---

## Task 20: Restore the glob, refresh docs, and green the gate

> **Amended 2026-10-22.** Parts of this task landed early, outside the plan:
>
> - Step 2: `AGENTS.md` and `CLAUDE.md` already describe rusticide;
>   `CLAUDE.md` is now a one-line `@AGENTS.md` import, so only `AGENTS.md`
>   (plus `README.md` and `CONTRIBUTING.md`) needs the rewrite.
>   `CHANGELOG.md` must NOT be reset to an unreleased 0.1.0: v0.1.0 was
>   published (a release of the placeholder binary). SP1 ships as 0.2.0,
>   bumped with `mise run release:prepare`.
> - Step 4: `crates-publish.yml` no longer exists. `publish.yml` (called
>   from `cicd.yml`) publishes with `cargo publish --workspace`, which
>   orders crates by dependency on its own, so no crate list is needed;
>   only `parity.yml` remains to reduce.
> - Step 5: `mise run ci` is unchanged, and `cicd.yml` runs
>   `mise run ci:quick` on every push, so a green CI run is the check.

**Files:**

- Modify: `Cargo.toml`
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: `AGENTS.md`, `CLAUDE.md`, `CONTRIBUTING.md`
- Modify: `.github/workflows/crates-publish.yml`
- Modify: `.github/workflows/parity.yml`
- Create: `docs/ARCHITECTURE.md`

**Interfaces:**

- Consumes: every crate from Tasks 2-19.
- Produces: a green `mise run ci`.

**Steps:**

- [ ] **Step 1: Restore the members glob**

```toml
exclude = ["crates/rusticide-fuzz"]
members = ["crates/*"]
```

Run: `cargo metadata --no-deps --format-version 1 >/dev/null`
Expected: exit 0.

- [ ] **Step 2: Rewrite the prose docs**

`README.md`, `AGENTS.md`, `CONTRIBUTING.md`, and `CLAUDE.md` mention
`rusticide` 24, 19, 13, and 5 times respectively. Rewrite each to describe
rusticide. Reset `CHANGELOG.md` to an unreleased section for 0.1.0.

- [ ] **Step 3: Write `docs/ARCHITECTURE.md`**

The crate graph and layering, derived from the design document's
"Architecture" section. Keep it under 200 lines and link to the design
rather than restating it.

- [ ] **Step 4: Fix the workflows**

`crates-publish.yml` names the old crates 29 times: replace the list
with the publish order, which is dependency order:
`rusticide-alloc`, `rusticide-core`, `rusticide-schema`,
`rusticide-dispatch`, `rusticide-io`, `rusticide-quantize`,
`rusticide-llm-tokenizer`, `rusticide-llm-sampling`, `rusticide`.

`parity.yml` targets GGUF inference that no longer exists in this
repository. Reduce it to the schema corpus round-trip until SP2 brings
the oracle.

- [ ] **Step 5: Run the full gate**

```bash
mise run fmt
mise run lint
mise run test
mise run ci
```

Expected: all green. Fix anything that is not.

- [ ] **Step 6: Verify the file-size constraint**

```bash
find crates -name '*.rs' -exec wc -l {} + | sort -rn | head -20
```

Expected: no file over 300 lines. Split any that are.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -S -m "docs: retarget the repository at rusticide

Restores the crates/* glob now that every member builds, rewrites the
prose docs, and reduces parity.yml to the schema corpus round-trip
until SP2 brings the oracle."
```

---

## Task 21: Squash the import commits into one

> **Cancelled 2026-10-22.** The branch was merged to `main` and pushed,
> `main` is public, and the signed, attested v0.1.0 release points at
> commits this task would rewrite. Squashing now means force-pushing a
> public branch and orphaning a release's provenance, which costs far more
> than the tidier history it buys. The steps below are kept for the record
> only; do not run them.

**This task rewrites history and MUST run last**, after Task 20 and after
every review is closed. It renumbers every commit on the branch, so
running it earlier invalidates the SHAs recorded in the ledger and in
each task's report.

The four commits at the base of the branch record how the repository was
assembled from the `xllm` reset, in an order that mattered while it was
happening and matters to nobody afterwards:

```text
4b481f9  chore: import the engineering scaffold
264aee6  chore: complete the scaffold import
a750001  docs: import licensing and community documentation
cfb98e0  chore: import the five surviving crates from xllm
```

Alongside them sits a repository-wide normalisation the human staged
separately: REUSE/SPDX headers applied by `comply annotate`, `hk`
upgraded from 1.50 to 1.56, TOML reformatting, and a new
`ignorefile.toml`. It is the same baseline work, done in a second pass.

Collapsing all five into one commit gives the branch a single honest
starting point instead of four bookkeeping steps plus an afterthought.

**Files:**

- Rewrites: the first four commits on `feat/foundation-sp0-sp1`
- Modify: the ledger at
  `.superpowers/sdd/<plan-basename>/progress.md`

**Interfaces:**

- Consumes: a clean working tree and no running agents.
- Produces: a branch whose first commit contains the full initial state.

**Steps:**

- [ ] **Step 1: Confirm nothing else is writing to the repository**

A rebase moves `HEAD` underneath anything holding it. Before starting:

- No implementer or reviewer subagent is running.
- No other editor session has this repository open. During SP1 a peer
  session named `rusticide-cd` was seen idle with it open; check for
  that class of thing, not just that one name.
- `git status` is clean, or whatever remains is stashed.

If any of these is unmet, **stop**. This task is not urgent and a
corrupted rebase is expensive.

- [ ] **Step 2: Take a recovery ref**

```bash
git branch backup/pre-squash-$(git rev-parse --short HEAD)
```

This is the undo. `git reset --hard` onto it restores everything if the
rebase goes wrong, and it costs nothing to leave behind.

- [ ] **Step 3: Squash the four**

```bash
git rebase -i --root
```

Keep `4b481f9` as `pick`; mark the next three `squash`. The editor then
offers the combined message -- replace it entirely with the message from
Step 5 rather than accepting the concatenation of four.

Note that `git rebase -i` is not available through the Bash tool in an
agent session. This step is the human's to run, or the agent scripts it
non-interactively with `git reset --soft` onto the root commit's parent
and a fresh commit.

- [ ] **Step 4: Fold in the staged normalisation**

If the 106-file normalisation is still staged rather than committed:

```bash
HK=0 git commit -S --amend --no-edit
```

If it was already committed as its own commit, mark that one `squash`
in Step 3 instead and skip this.

- [ ] **Step 5: The message**

```text
init: initialise the rusticide repository

Assembles the project from the xllm reset: the engineering scaffold
(toolchain pins, lint configuration, CI workflows, supply-chain policy,
mise tasks), dual MIT/Apache-2.0 licensing with REUSE compliance, the
contributor and security policies, and the five crates that survived the
reset -- GGUF reading, tokenizer, quantization, sampling, and the fuzz
targets.

Includes the repository-wide baseline applied afterwards: SPDX headers
via comply annotate, hk 1.50 -> 1.56, TOML reformatting, and
ignorefile.toml.

The workspace does not resolve at this commit; Task 1 repairs it.
```

That last line matters. It was true of `cfb98e0` and stays true of the
squashed commit, and a reader who checks out this commit and finds a
broken workspace deserves to know it is expected.

- [ ] **Step 6: Verify the tree is byte-identical**

A squash must not change content. Compare the rewritten branch's tree
against the recovery ref:

```bash
git diff backup/pre-squash-<sha> --stat
```

Expected: **no output**. Any difference means the rebase dropped or
altered something, and the recovery ref is right there.

- [ ] **Step 7: Re-run the gates**

```bash
mise run test
hk check --all
```

Both must pass, as they did before the rewrite.

- [ ] **Step 8: Rewrite the ledger's SHAs**

Every `Task N: complete (commits X..Y)` line now names commits that no
longer exist. Map each to its rewritten equivalent with
`git log --format='%h %s'` and correct them, or replace the SHA ranges
with a note that they predate the squash. Leaving them stale makes the
ledger actively misleading -- it would point at objects git cannot
resolve.

- [ ] **Step 9: Delete the recovery ref**

Only once Steps 6 and 7 have both passed:

```bash
git branch -D backup/pre-squash-<sha>
```

---

## Self-Review

**Spec coverage.** Every SP0 item maps to Tasks 1, 17-20; every SP1 item
maps to Tasks 2-16. The spec's SP1 acceptance criteria appear as
executable tests: the 2,585-schema round-trip in Task 13, and `add`
through the real table in Task 16. The `unsafe` policy from the spec is
established in Task 5 with `docs/UNSAFE.md` and
`clippy::undocumented_unsafe_blocks`. The aliasing decision
(`Arc<RwLock<Storage>>`) is Task 6, and its self-aliasing guard is
Task 15. The layering inversion found during the reset is fixed in
Tasks 17 and 18.

Not covered here, by design: `rusticide-cpu`, `rusticide-gpu`,
`rusticide-kernels`, `rusticide-simd`, `rusticide-autograd`, the oracle,
and the codegen emitter. Those are SP2 onward and get their own plans.

**Type consistency.** `Error` is the core's error type throughout; the
schema crate uses its own `ParseError` because it has no dependency on
the core. `DispatchKeySet::table_index` is used by `OperatorEntry` in
Task 16 exactly as defined in Task 8. `Storage::same_buffer_as` (Task 6)
backs `Tensor::shares_storage_with` (Task 9), which backs
`distinct_storages` (Task 15).

**Known follow-up.** Task 6 requires `Deref`/`DerefMut` on
`rusticide_alloc::Buffer` so the `RwLock` guards index bytes directly;
that is called out in Task 6 Step 3 and must be implemented with tests
in `rusticide-alloc` at that point.
