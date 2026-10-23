# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2026-10-22

### Features

- **schema:** add FunctionSchema and corpus round-trip
- **dispatch:** add IValue and Stack
- **dispatch:** add KernelFunction, aliasing guard
- **dispatch:** add Dispatcher and OperatorEntry
- **core:** add element-access error variants and Element
- **core:** build tensors from slices and read/write elements
- **core:** add deep_copy and the EagerOps reshape/contiguous
- **core:** add eager broadcasting add and 2-D matmul
- **io:** add the GGUF value layer
- **io:** parse GGUF headers, metadata and tensor infos
- **io:** materialize GGUF tensors, dequantizing quantized types
- **ggml:** port the math operators onto rusticide-core
- **rusticide:** add the facade crate with an embedded CLI
- **dispatch:** let kernels return errors
- **core:** expose ordered paired storage reads

### Bug Fixes

- **mise:** build cargo tools without cargo-binstall
- **schema:** reject ")" before "(" in annotations
- **core:** saturate Shape::numel on overflow
- **core:** saturate contiguous strides
- **core:** type-enforce storage length and fallible eager copies
- **quantize:** allocate dequantized output fallibly
- **core:** acquire paired read locks in a global order
- **sampling:** reject non-finite logits in sample_with_temp
- **ggml:** recompute the graph on every forward
- **tokenizer:** reject non-finite scores from GGUF metadata
- **ci:** run the nightly fuzz job through mise
- **sampling:** order logits with total_cmp and map NaN to -inf
- **tokenizer:** build the piece lookup once and bound token length
- **tokenizer:** walk a compact arena trie in linear time
- **ggml:** size GetRows rows with saturating shape math
- **io:** reject duplicate metadata keys
- **tokenizer:** drop the unused Trie Default impl
- **tokenizer:** binary-search sorted trie edges
- **mise:** resolve the lockfile online in release:prepare
- **sampling:** reject non-finite logits in greedy sampling

### Refactor

- **quantize:** rebind to rusticide-core and split by format
- **sampling:** rebind to rusticide-core and split by concern
- **ggml:** rebind the compute graph to rusticide-core and split
- **tokenizer:** read GGUF metadata from rusticide-io, drop Model
- **fuzz:** retarget the GGUF targets at rusticide-io

### Documentation

- **plan:** amend tasks 19-21 for the shipped release
- record the Tasks 17-18 re-plan outcome
- retarget the repository at rusticide
- correct the EagerOps seam and purge stale references
- fix intra-doc links and document trusted-API and locking caveats

### Testing

- **dispatch:** prove slot indexing and entry reuse
- **core:** pin matmul and broadcast index order
## [0.1.0] - 2026-10-22

### Features

- **core:** add DType with promotion rules
- **core:** add Device, Layout and Error types
- **core:** add Shape with broadcasting and Strides
- **alloc:** add Buffer and establish the unsafe policy
- **core:** add Storage with shared mutable buffers
- **core:** add DispatchKey and BackendComponent
- **core:** add DispatchKeySet
- **core:** add Tensor, TensorImpl and VersionCounter
- **schema:** add Type with round-trip-asserted parsing
- **schema:** add Annotation with the exclusivity rule
- **schema:** add Argument, Arguments, Return and OperatorName
- **cli:** add the rusticide binary

### Bug Fixes

- **core:** make DType promotion commutative
- **alloc:** restore the full workspace lint set in rusticide-alloc
- **core:** stop AllocationFailed lying about overflowed sizes
- **core:** replace loose backend-count assertion with exact compile-time check
- **core:** close DispatchKeySet's dispatch-safety gaps
- **core:** drop speculative Tensor::storage_offset getter
- **schema:** DRY BaseTy's string table, cover every variant in tests
- **schema:** make BaseTy table-completeness a compile-time guarantee
- **mise:** drop glaucus feature from coverage run
- **mise:** skip string deps in version:sync
- **ci:** publish only the tagged commit
- **ci:** narrow codeql.yml to what cicd.yml grants
- **ci:** run the gitleaks CLI instead of the action
- **mise:** install every tool from prebuilt binaries
- **mise:** checksum every prebuilt tool binary

### Refactor

- **schema:** split BaseTy into its own module
- rename xllm imports to rusticide
- **schema:** drop redundant empty alias check
- **schema:** share the round-trip check

### Documentation

- import licensing and community documentation
- add the rusticide framework design and foundation plan
- repair planning documents after the global rename
- stop the plan's SPDX example from tripping comply
- correct the SPDX header the plan tells every task to write
- point plan links at the framework design
- **plan:** use rusticide crate names
- make AGENTS.md the single source of truth
- describe the CI/CD pipeline and release flow

### Styling

- apply rustfmt to rusticide-core

### Testing

- **schema:** give BaseTy::parse direct test coverage
- **schema:** cover the before/after multi-letter alias exclusivity
- **alloc:** cover Buffer::is_empty
- **core:** cover the remaining getters and paths
- **schema:** cover rejection and guard paths

### Miscellaneous Tasks

- import the engineering scaffold
- complete the scaffold import
- import the five surviving crates from xllm
- repair workspace after reset
- restore correct exclude path for rusticide-fuzz
- **license:** add SPDX headers across the repo
- lower MSRV to 1.88 and track Cargo.lock
- move build.rs into crates/rusticide
- **mise:** import the task suite and hk hooks
- bump toolchain to 1.99 and dirs to 7
- **mise:** remove glaucus-only tasks
- **claude:** track settings and project skills
- **license:** add SPDX headers to the Claude files
- bump action pins and run the pin checker via uv
- **rust-env:** adopt glaucus's setup-rust-env composite
- centralise cargo profiles in .cargo/config.toml
- regenerate .gitignore; taplo skips build output
- **mise:** add release tasks with cosign and syft
- replace ci.yml with a glaucus-style CICD pipeline
- publish signed, attested multi-platform releases
- **taplo:** skip the machine-local mise.local.toml

<!-- generated by git-cliff -->
