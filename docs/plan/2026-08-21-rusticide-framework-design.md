<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Rusticide -- Pure-Rust Deep Learning Framework (Design)

Status: Approved design, ready for implementation planning.
Date: 2026-08-21.
Owner: Elio Severo Junior.

## Context: what this document changes

This repository began as `xllm`, an LLM inference engine. Its roadmap
(`docs/plan/XLLM_PLAN.md`, Layers 0-4) targeted running the models vLLM
serves and beating vLLM on benchmarks.

This document repositions the project. Rusticide is a **deep learning
framework** -- a pure-Rust reimplementation of what PyTorch does -- and
the LLM work becomes an application layer on top of it rather than the
product itself.

The repositioning was chosen deliberately after measuring PyTorch's
actual architecture (see **Evidence base** below). It is not a pivot
away from the LLM goal: a framework that can train is a strictly larger
capability than an engine that can only infer, and the existing
inference work survives as its first real consumer.

### What happens to the existing Layer 0-4 roadmap

- **Layer 0 (correctness foundation, GGUF token-id parity)** stays
  valuable and moves into the LLM application layer. Its discipline --
  prove correctness against a reference before measuring anything -- is
  adopted wholesale by this design as the oracle strategy.
- **Layers 1-3 (HuggingFace formats, SIMD/threading, serving)** become
  application-layer concerns that consume the framework.
- **Layer 4 (GPU)** is absorbed into sub-project 4 and, importantly,
  **no longer requires an exception to `unsafe_code = "deny"`**. See
  "The no-unsafe rule survives GPU" below.

## Goal

Rusticide trains and runs neural networks in pure Rust, on CPU and GPU,
using PyTorch as the numerical reference.

Concretely, "success" means:

- Models train from scratch: autograd, optimizers, the full loop.
- Every implemented operator is verified against PyTorch's output within
  a documented per-operator tolerance.
- The same model produces equivalent results on CPU and GPU.
- The LLM application layer runs on top of the framework.

## Non-goals, and why

"Everything PyTorch has" is not literally achievable under the pure-Rust
constraint. Three PyTorch subsystems are excluded, and the reason in each
case is that they are not porting problems at all:

- **TorchDynamo** (129,491 lines) installs a CPython frame-evaluation
  hook and rewrites Python bytecode in flight. With no Python
  interpreter there are no frames to intercept; the subsystem's purpose
  does not exist here. A Rust equivalent would be a proc-macro or
  trace-based JIT -- a different design that should carry a different
  name.
- **TorchInductor** (282,513 lines) emits Triton, which is a Python DSL,
  and C++ source at runtime. A pure-Rust code generator would target
  WGSL or SPIR-V instead. Again: not a port.
- **Symbolic shapes** (11,702 lines) are a sympy-backed computer-algebra
  system reached from C++ by acquiring the GIL on every symbolic
  operation. Rust has no mature sympy equivalent.

**TorchScript** (213,412 lines of C++) is excluded for a different
reason: it is superseded by `torch.compile` and `torch.export` upstream,
so porting it would be archaeology.

Deferred rather than excluded: distributed training, quantization-aware
training, and true aliasing views (see "Views and functionalization").

## Hard constraints

These are not preferences. They shape every decision below.

1. **Pure Rust.** No C, no C++, no Python, no CPython. No `pyo3`, no
   `libtorch` bindings, no `cc`/`bindgen` of vendored C sources, no
   BLAS, no cuDNN, no NCCL, no Triton.
2. **`unsafe_code = "deny"`** at the workspace level, inherited from the
   existing configuration and retained as the default. Crates outside
   the framework core may opt out under the policy in "Where `unsafe` is
   permitted" below.
3. **No `unwrap`/`expect` in production code** (`clippy::unwrap_used`,
   `clippy::expect_used`), permitted in `#[cfg(test)]`.
4. **`-D warnings`** in `.cargo/config.toml`, escalating every rustc and
   clippy warning into a build failure.
5. **Files stay under 300 lines**, per the project's engineering
   standards. This is a change from the current state: several crates
   are a single `lib.rs` of 1,100-1,800 lines.

### Where the pure-Rust line actually falls

Every GPU driver exposes a C ABI -- CUDA is `libcuda.so`, Vulkan is
`libvulkan.so`, Metal is an Objective-C runtime. "Zero C" and "has a GPU"
cannot both hold literally. Rust's own `std` links libc.

The practical line adopted here: **no C or C++ source in our tree, no
vendored C libraries, no `bindgen` of third-party C headers.** System
drivers reached through a safe Rust abstraction are acceptable.

GPU access therefore goes through `wgpu`, with kernels authored in WGSL
and compiled by `naga` -- both pure Rust. `rust-gpu` (Rust compiled to
SPIR-V) is a desirable upgrade but historically requires a pinned
nightly toolchain, so it is a candidate to evaluate in sub-project 4,
not a premise.

The oracle fixture generator is written in Python and lives outside the
Rust tree (`tools/oracle/`). This does not violate the constraint: it
runs offline, its outputs are versioned data files, and the published
crate has zero Python at runtime or build time.

### The no-unsafe rule survives GPU

The previous roadmap recorded that Layer 4 would need "an explicit,
documented exception to the core no-unsafe rule" because GPU implies
`unsafe`/FFI.

Choosing `wgpu` removes that dilemma. `wgpu` exposes a safe Rust API;
whatever `unsafe` it needs is internal to the dependency.
`unsafe_code = "deny"` is a lint on this workspace's own code and is not
transitive. **No exception is required for GPU.**

### Where `unsafe` is permitted

`unsafe` is permitted outside the framework core, in a crate of its own,
with a written justification. It is never permitted in `rusticide-core`,
`rusticide-dispatch`, `rusticide-schema`, `rusticide-codegen`, or
`rusticide-ops`.

A crate that opts out must satisfy all of the following:

1. **Isolation.** The `unsafe` lives in a dedicated crate whose reason
   for existing is that capability. It is never mixed into a crate that
   also carries ordinary logic.
2. **A safe public API.** No `unsafe` in the signatures callers touch.
   The crate's job is to convert an unsafe capability into a safe one.
3. **A `// SAFETY:` comment on every block**, stating the invariant that
   makes the operation sound -- not restating what the code does.
4. **A justification in the crate README**: what is being done, why safe
   Rust cannot express it, and what would have to change for the crate
   to be deleted.
5. **Registration in `docs/UNSAFE.md`**, a single index of every
   `unsafe`-permitted crate. If the list is not short, the policy has
   failed.
6. **Miri in CI** for the crate's test suite, plus fuzzing where it
   parses untrusted input.

Mechanically, the workspace lint table stays `unsafe_code = "deny"` and
a permitted crate declares its own `[lints.rust]` instead of inheriting
with `[lints] workspace = true`.

The anticipated candidates, and none is speculative:

- **`rusticide-simd`** -- stable Rust reaches AVX2, AVX-512, and NEON
  only through `core::arch` intrinsics, which are `unsafe`. This is the
  largest and most legitimate use, and it keeps `rusticide-cpu` itself
  safe by absorbing the intrinsics behind typed vector operations.
- **`rusticide-alloc`** -- aligned allocation and, more importantly,
  uninitialized buffers. Zero-filling every tensor allocation is a real
  and unnecessary cost on a framework that immediately overwrites most
  of them; avoiding it needs `MaybeUninit` and the raw allocator API.
- **`rusticide-io`** -- memory-mapped files. `memmap2` is inherently
  unsafe because the backing file can change underneath the mapping, and
  the existing GGUF reader already depends on it.

This arrangement is dependency inversion, not a loophole:
`rusticide-core` declares the `Allocator` trait and stays safe, while
the implementation that needs `MaybeUninit` is injected from
`rusticide-alloc`. The core never sees an `unsafe` block, and the crates
that do are small enough to audit in one sitting.

## Evidence base

The measurements below were taken on `pytorch/pytorch @ 43f8f31f6a2`
(2026-08-21), first-party code only, excluding tests, `third_party/`,
and build output. They are the reason the architecture takes the shape
it does.

- Total first-party: 2,486,249 lines. `torch/` alone is 1,269,885 lines
  of Python against 528,319 of C++ -- PyTorch is roughly half Python.
- `native_functions.yaml`: 16,239 lines declaring **2,585 operator
  overloads** across 1,554 base names.
- `derivatives.yaml`: 3,299 lines, **688 gradient formulas**.
- `torchgen/`: 26,437 lines of Python, generating **937,056 lines** of
  C++. About a third of "PyTorch the C++ library" is machine-written,
  and the schemas that drive it are language-agnostic.
- Hand-written kernels in `aten/src/ATen/native/`: 426,153 lines, of
  which a large share delegates to vendor libraries (75 files to
  oneDNN/MKLDNN, 51 to cuDNN, 17 to cuBLAS).

Two structural findings drive the design more than the sizes do:

**The dispatcher is two instructions.** A `DispatchKeySet` is one
`u64`, split into roughly 16 low "backend" bits and 48 high
"functionality" bits, stored orthogonally to avoid the quadratic blowup
of one bit per (backend x functionality) pair. Kernel selection is:

```text
functionality_idx = clz(repr >> num_backends)
(offset, mask)    = offsets_and_masks[functionality_idx]
backend_idx       = clz((repr & mask) >> 1)
table[offset + backend_idx]
```

Priority is encoded purely in numeric enum order -- nothing declares
that Autograd outranks Dense, the variant simply sits higher. The
ordering in `c10/core/DispatchKey.h` is normative specification.

**The schema language already carries borrow annotations.**
`Tensor(a!)` means "belongs to alias set `a`, and is written".
`torchgen/model.py` enforces this at parse time:

```python
if is_write and len(alias_set) > 1:
    raise AssertionError("alias set larger than 1 is not mutable")
```

A mutable reference may belong to exactly one alias set -- aliasing XOR
mutation, Rust's own exclusivity rule, declared in a DSL and enforced
across all 2,585 schemas. Counting the annotations:

- 1,513 (58.5%) carry no annotation: pure, fresh return.
- 981 (38%) carry a mutable annotation: a single `&mut`.
- 91 (3.5%) return a read-only aliasing view.
- 14 use the wildcard `(a -> *)`: the `split`/`chunk`/`unbind` family.

96.5% of the operator surface is already expressible in safe Rust with
no runtime machinery.

## Architecture

### Crate graph

The workspace follows the facade pattern established in
[glaucus](https://github.com/elioseverojunior/glaucus): one facade crate
that re-exports the public API and embeds the CLI behind a feature gate,
with the engine in a `-core` crate beneath it.

**Facade.**

- `rusticide` -- public API re-exports plus the CLI. Carries both
  `src/lib.rs` and `src/main.rs`, with
  `[[bin]] required-features = ["cli"]` so library consumers never pull
  `clap` or `tracing`. Only `rusticide-core` is a non-optional
  dependency; everything else sits behind `dep:` feature gates.

**Framework core.**

- `rusticide-core` -- `DType`, `Device`, `Layout`, `Shape`, `Strides`,
  `Storage`, the `Allocator` trait, `DispatchKey`, `DispatchKeySet`,
  `Tensor`/`TensorImpl`, the version counter, and `Error`.
- `rusticide-dispatch` -- `Dispatcher`, `OperatorEntry`,
  `KernelFunction`, `IValue`, `Stack`, and the boxed calling convention.
- `rusticide-schema` -- the `torchgen` data model ported to Rust:
  `FunctionSchema`, `Type`, `Annotation`, `Argument`, `Arguments`,
  `OperatorName`, with round-trip assertions.
- `rusticide-codegen` -- build-time emitter consuming `rusticide-schema`.
- `rusticide-ops` -- the generated operator surface and its registration.

`Tensor` lives in `rusticide-core` rather than its own crate because
`Dispatcher` needs `Tensor` (an `IValue` can hold one) and `Tensor`
needs `DispatchKeySet`. That is a dependency cycle. PyTorch resolves it
the same way: `TensorImpl` in `c10`, `IValue` in `ATen/core`.

**Backends.**

- `rusticide-cpu` -- the `TensorIterator` equivalent (broadcasting,
  dtype promotion, vectorized loops), GEMM in pure Rust, `rayon`.
- `rusticide-gpu` -- the `wgpu` backend: adapters, buffers, submission.
- `rusticide-kernels` -- WGSL kernel sources and their bindings.

**Low-level capability crates.** These are the only crates permitted to
carry `unsafe`, under the policy in "Where `unsafe` is permitted". Each
exists solely to convert an unsafe capability into a safe API, and each
is small enough to audit in one sitting.

- `rusticide-simd` -- AVX2, AVX-512, and NEON via `core::arch`.
- `rusticide-alloc` -- aligned and uninitialized buffer allocation,
  implementing the `Allocator` trait declared in `rusticide-core`.

**Training and application.**

- `rusticide-autograd` -- `Node`, `Edge`, `SavedTensor`, the
  dependency-counting engine, and VJPs generated from `derivatives.yaml`.
- `rusticide-nn` -- modules.
- `rusticide-optim` -- optimizers.
- `rusticide-io` -- safetensors, GGUF (from the deleted `rusticide-model`
  reader), and a pickle VM for `.pt` checkpoints.
- `rusticide-ggml` -- the legacy GGML-style compute graph, rebound to
  `rusticide-core` and kept as is until the framework replaces it.
- `rusticide-llm-*` -- the former `rusticide-*` crates: model, tokenizer,
  sampling, context, quantize.

**Development.**

- `rusticide-oracle` -- fixture format and comparison harness
  (dev-dependency).
- `rusticide-bench` -- criterion benchmarks as their own crate.
- `rusticide-fuzz` -- fuzz targets, excluded from the workspace because
  `libfuzzer-sys` needs nightly and cannot satisfy
  `unsafe_code = "deny"`.

### CLI surface

Subcommands follow the `src/cli/cmd/*.rs`-per-command layout from
glaucus:

- `run` -- execute a model.
- `inspect` -- inspect a checkpoint (safetensors, GGUF, `.pt`).
- `convert` -- convert between checkpoint formats.
- `devices` -- enumerate `wgpu` adapters.
- `schema` -- query and validate operator schemas; dogfoods
  `rusticide-schema`.
- `oracle` -- run verification against PyTorch fixtures.
- `bench` -- benchmark individual operators.
- `completions`, `version` -- standard.

## Core mechanisms

### Dispatcher

`DispatchKeySet` is a newtype over `u64` reproducing PyTorch's bit
layout exactly, because the layout and the enum ordering together are
the specification for kernel selection. `leading_zeros()` replaces
`clz`. `OperatorEntry` holds a flat array of `KernelFunction` indexed by
the computed runtime key.

Alias keys (`CompositeImplicitAutograd` and friends) are not bits. They
are resolved at registration time into concrete table slots with the
precedence `CompositeExplicitAutogradNonFunctional >
CompositeExplicitAutograd > CompositeImplicitAutograd > Autograd`.

`IValue` becomes a Rust `enum`. This is one of the places the port
improves on the original: PyTorch hand-rolls a 27-variant tagged union
with a manual `Tag` field and hand-written payload management, and Rust
gets exhaustive matching for free.

### Eager ops and the SP2 removal seam

The legacy eager API ported in SP1 (`from_slice`, `get`, `set`, `add`,
`matmul`, `reshape`, `contiguous`, `deep_copy`) conflicts with the
dispatcher design: it computes directly instead of routing through the
dispatch table. It is isolated behind the `EagerOps` trait in
`rusticide-core` so SP2 can delete it in one place and re-express the
callers as dispatched operators. It is not re-exported from the facade.

### Deferred from SP1

- **Allocation ceiling.** Outputs that amplify their inputs (`[m,0] x
  [0,n]`, `[N,1] + [1,M]`) are allocated fallibly, but zero-fill can still
  trigger an overcommit OOM-kill on Linux. A ceiling is policy for the SP2
  allocator and its `alloc_zeroed` path.
- **GGUF metadata amplification.** `GgufValue` arrays cost about 24-32
  bytes per element, so a hostile file can expand its metadata that much
  in memory. Typed array variants or a size cap belong to SP2.
- **`try_vec_with_capacity`.** It is `pub` but doc-hidden, and is removed
  once SP2 decodes directly into `Tensor` storage.

### Schema model and code generation

`rusticide-schema` ports `torchgen/model.py`'s data model: immutable
types that parse from and print back to the JIT schema string.

The acceptance criterion is inherited from the original. PyTorch's
parsers each end with a round-trip assertion:

```python
r = FunctionSchema(...)
if str(r) != func:
    raise AssertionError(f"{str(r)} != {func}")
```

The Rust port must parse all 2,585 schemas from PyTorch's own
`native_functions.yaml`, re-emit each one, and assert byte equality.
This is a milestone with an unambiguous pass condition, and it delivers
the alias annotations as a side effect.

Code generation is staged rather than up-front. The parser lands early
because it is self-grading; roughly 20 operators are then written by
hand to prove the runtime end to end; only then is the emitter switched
on for the long tail. Writing the full generator before the first
addition works is a way to spend months without a running program.

### Aliasing and mutation

The obvious safe-Rust design is `Arc<Storage>` with `Arc::make_mut` --
copy-on-write for free, mirroring the COW work PyTorch itself added in
`c10/core/impl/COW.cpp`.

**It breaks training.** In a training loop the parameter is shared
between the module and the optimizer. `p.add_(grad)` under
`Arc::make_mut` clones the storage, and the module never observes the
update. That is not an inefficiency; it is incorrect.

The real requirement is that parameters be **shared and mutable**, and
`unsafe_code = "deny"` closes PyTorch's own route.

The relaxation in "Where `unsafe` is permitted" does not reopen this.
`Storage` lives in `rusticide-core`, which is never permitted to carry
`unsafe`, so `UnsafeCell` over shared storage is unavailable by design
rather than by oversight. That is the intended reading: the core's
soundness should not rest on a hand-written argument.

**Decision: `Arc<RwLock<Storage>>`.** Kernels acquire the lock once per
launch, not per element -- roughly 20ns against a kernel measured in
microseconds. In exchange the framework keeps PyTorch's semantics, which
is what makes the oracle worth having.

One hazard must be handled explicitly. `a.add_(a)` requires a write and
a read on the same storage and would deadlock. The dispatcher compares
storage pointers across arguments before locking and degrades to a
single write lock when they coincide. This check is part of the boxed
calling convention, not left to individual kernels.

Alternatives considered and rejected:

- **Value semantics with functional updates** (`let (model, opt) =
  opt.step(model, grads)`), as `burn` does. Safe, idiomatic, no locks,
  no deadlock hazard -- but diverges from PyTorch's API, which weakens
  the oracle's value.
- **A central parameter arena** with modules holding handles. The most
  efficient and most statically checkable option, but the largest API
  divergence and awkward under dynamic graphs.

### Views and functionalization

Views ship in two tiers.

**Tier 1 (v1): value semantics.** `view` behaves as `view_copy`. A view
does not alias its base, so `b = a.view(...); b.add_(1)` leaves `a`
unchanged. This is precisely what PyTorch's own functionalization pass
produces for backends that cannot alias, and its design note names the
audience directly:

> This is useful for backends that don't support aliasing, like XLA and
> Vulkan. It's also necessary in order to remove mutation from a
> program.

PyTorch supports this configuration with 54 generated `*_copy` operator
variants. A Rust backend wanting value semantics is the case that
subsystem was written for.

**Tier 2 (deferred): true aliasing views.** Requires runtime borrow
tracking on storage. Affects 91 of 2,585 schemas plus the 14 wildcard
`split`/`chunk`/`unbind` entries.

The divergence in tier 1 is documented behaviour, not an accident, and
it is scoped: it changes program semantics for code that mutates through
views, and changes nothing about the numerics of any individual
operator. The oracle tests operator numerics, so it remains valid.

## The oracle

Pure Rust means PyTorch cannot be called at test time. The reference is
therefore recorded fixtures.

**Generation.** A Python script under `tools/oracle/`, outside the Rust
tree, reads a manifest of operator plus input specifications, evaluates
them under PyTorch, and writes fixtures. It runs offline or in a
dedicated CI job; it is never part of a `cargo build`.

**Determinism.** Inputs come from a seeded, documented RNG so fixtures
are reproducible. Tensors are kept small -- 64 elements or fewer -- so
the corpus stays versionable.

**Format.** safetensors. It reads in pure Rust, and `rusticide-io` needs
that reader regardless, so the oracle dogfoods production code.

**Metadata per fixture.** dtype, device, PyTorch version, and a
per-operator tolerance. Tolerance is not global: GEMM needs more slack
than elementwise because summation order differs; `f16` and `bf16` need
far more than `f32`. Bitwise equality is not a goal and is not
achievable with a different GEMM.

**Coverage as a metric.** The share of the 2,585 schemas with fixtures
is tracked and reported. It is the project's honest progress number.

## Testing strategy

Seven layers, each catching a distinct class of defect:

- **Schema round-trip** -- an incomplete parser. Self-grading: parse all
  2,585 schemas, re-emit, require byte equality.
- **Oracle** -- wrong numerics, per operator, dtype, and device.
- **Gradient check** -- wrong derivative formulas, via finite
  differences against the analytical VJP. Requires no fixtures and no
  PyTorch, which makes it the cheapest high-value test in the suite and
  the primary guard on all 688 formulas.
- **Device parity** -- kernel bugs, by comparing CPU against GPU within
  tolerance.
- **Property tests** (`proptest`) -- broadcasting rules, dtype
  promotion, stride computation, shape inference.
- **Fuzzing** -- format parsers (GGUF, safetensors, pickle) consume
  untrusted input. The existing `-fuzz` crate covers this.
- **Miri** -- undefined behaviour in the `unsafe`-permitted crates
  (`rusticide-simd`, `rusticide-alloc`, `rusticide-io`). Required in CI
  for those crates and only those; Miri cannot execute SIMD intrinsics
  or memory maps, so those paths need scalar fallbacks exercised under a
  `cfg(miri)` branch, which is itself a useful discipline.
- **Integration and benchmarks** -- end-to-end model execution and
  criterion regression tracking.

## Sub-project decomposition

This design is too large for a single implementation plan. It decomposes
into sub-projects, each getting its own spec, plan, and implementation
cycle. This document designs the whole; only SP0 and SP1 are planned
immediately.

- **SP0 -- Rename and restructure.** `rusticide-*` becomes `rusticide-*`;
  package metadata, git remote, and documentation follow. The facade
  crate is created. Single-file crates are split into modules under 300
  lines. *Done when:* `cargo build`, clippy, and CI are green under the
  new name and no source file exceeds 300 lines.

  The repository was reset on 2026-08-21: build artifacts, the crates
  coupled to the old tensor type, and the superseded planning documents
  were deleted (all preserved in the original `xllm` repository, which
  retains its history and remote). Five crates survive -- `rusticide-ggml`,
  `rusticide-tokenizer`, `rusticide-quantize`, `rusticide-sampling`, `rusticide-fuzz`,
  totalling 4,061 lines -- and the workspace does not currently resolve.
  Repairing it is SP0's first task, and the damage is exactly eight
  `use` lines:

  > **Amended 2026-10-22.** The "eight `use` lines" estimate was wrong:
  > the old `Tensor` API had to be ported into `rusticide-core`. See Tasks
  > 17-18 in the foundation plan.

  - `rusticide-ggml`, `rusticide-sampling`, `rusticide-quantize` import `Tensor`,
    `DType`, and `TensorError` from the deleted `rusticide-tensor`. These
    rebind to `rusticide-core` once it exists, so these three crates are
    blocked on SP1 rather than repairable in isolation.
  - `rusticide-tokenizer` and the fuzz targets import `Model` and
    `GGUFValue` from the deleted `rusticide-model`. A tokenizer depending on
    a model crate is a layering inversion; `GGUFValue` is format
    metadata and belongs in `rusticide-io` alongside the GGUF reader.
    Moving it there fixes the inversion rather than reproducing it, and
    unblocks the tokenizer without waiting on SP1.
- **SP1 -- Foundation.** `rusticide-core`, `rusticide-schema`,
  `rusticide-dispatch`, and `rusticide-alloc` -- `Storage` cannot exist
  without an allocator, and this is where the `unsafe` policy gets its
  first test case and `docs/UNSAFE.md` its first entry. *Done when:* all
  2,585 schemas round-trip byte exactly, and `add` dispatches through
  the real table to a hand-written CPU kernel.
- **SP2 -- CPU tier 1 and the oracle.** The `TensorIterator`
  equivalent, `rusticide-simd`, GEMM, roughly 100 operators, and the
  fixture pipeline. SIMD lands here rather than later because a GEMM
  benchmark without it measures nothing, and the risk it retires is the
  project's largest. *Done when:* those operators match PyTorch within
  their declared tolerances.
- **SP3 -- Autograd.** The engine and derivative codegen. *Done when:*
  gradient checks pass and an MLP trains on CPU.
- **SP4 -- GPU.** The `wgpu` backend and WGSL kernels. *Done when:* the
  same MLP trains on GPU with matching numerics. `rust-gpu` is evaluated
  here.
- **SP5 -- nn, optim, and the LLM layer.** *Done when:* the LLM
  application runs on the framework core.
- **SP6 -- Operator surface expansion.** Code generation switched on for
  the long tail.
- **SP7 -- Distributed and compilation.** Re-conceived rather than
  ported; deliberately last.

## Risks

- **GEMM performance.** Losing BLAS means matrix multiplication is our
  code. `faer` and `gemm` are credible pure-Rust options but do not
  reach MKL or cuDNN. This is the single largest performance risk and
  should be benchmarked in SP2 before the surface grows.
- **Convolution and attention.** The two hardest kernels to write well,
  and cuDNN is unavailable. Expect these to dominate SP4.
- **`rust-gpu` toolchain.** Requires a pinned nightly historically.
  Mitigated by treating WGSL plus `naga` as the baseline and `rust-gpu`
  as an upgrade to evaluate.
- **Prior art.** `burn` with `cubecl` made substantially the same bet --
  pure Rust, portable GPU. Reviewing what they got wrong is a mandatory
  input to SP1, not optional reading. `candle` is worth the same review
  for its pickle VM and format handling.
- **Fixture corpus size.** 2,585 schemas times several cases could
  outgrow a plain git repository. Mitigated by the 64-element cap;
  revisit with LFS if it does not hold.
- **Scope.** The framework is a multi-year effort at this scale. The
  sub-project boundaries exist so that each one delivers something
  verifiable on its own.

## References

Files in `pytorch/pytorch @ 43f8f31f6a2` that this design draws on
directly:

- `c10/core/DispatchKey.h`, `c10/core/DispatchKeySet.h` -- the bit
  layout and priority ordering.
- `c10/core/TensorImpl.h` -- the field budget discipline (208 bytes on
  64-bit, enforced by `static_assert`).
- `c10/core/impl/README-cow.md` -- the copy-on-write thread-safety
  argument.
- `aten/src/ATen/core/dispatch/OperatorEntry.cpp` -- alias key
  precedence resolution.
- `aten/src/ATen/FunctionalTensorWrapper.h` -- Note
  [Functionalization Pass In Core].
- `aten/src/ATen/TensorIterator.h` -- broadcasting and type promotion.
- `torchgen/model.py` -- the schema data model and its round-trip
  assertions.
