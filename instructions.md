---
name: instructions.md
created_by: elioseverojunior@gmail.com
---

<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Instructions

`rusticide` is a pure-Rust deep learning framework modelled on PyTorch's
design (see README.md and docs/plan/2026-08-21-rusticide-framework-design.md).
I want to focus on running on CPU, and it must be fast.
As a later feature I want to enable GPU support (AMD and NVIDIA GPUs), so
we need a design that can handle this.

## Rust SDK Requirements

- Must use Rust (stable), on a channel that allows us to use new features.
- We must focus on security-first design and performance.

## AI Requirements

You must write the needed agents, skills, prompts, and other configuration,
including the agentic tooling.
I want AI-agnostic configuration that can be reused by Claude, Codex, Copilot,
Bedrock, and others.
I want the AI to be able to do fully autonomous work.
Check [docs/guidelines/contribution.md](docs/guidelines/contribution.md).

## Coding Specifications

Ensure we use TDD before writing any line of code.
Ensure these development principles:

1. The codebase must have 100% coverage.
2. TDD (if we need to change existing code that has no tests, first write the
   test using TDD and ensure the test is working, then start to refactor the
   code we intended to modify, always following the TDD implementation loop:
   red -> code -> test, until green.)
3. KISS (Keep It Simple, Stupid)
4. DRY (Don't Repeat Yourself)
5. YAGNI (You Aren't Gonna Need It)
6. TDA (Tell Don't Ask)
7. SOLID (use the SOLID principles that make sense for the project).
