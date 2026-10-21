<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Agent session notes (July 2026, pre-reset)

Moved out of `AGENTS.md`. These notes describe the llama.cpp-port codebase
that existed before the repository was reset into the current framework
(see `docs/plan/2026-08-21-rusticide-framework-design.md`). The code they
mention (`rusticide-cli download`, GGUF Q2_K handling) no longer exists.
Kept for the licensing rationale and project history only.

## Session context (Jul 14, 2026)

### What was done

- **Download subcommand**: Added `rusticide-cli download` to fetch GGUF models from HuggingFace Hub using reqwest.
- **Model caching**: Models saved to `~/.cache/huggingface/hub/` following HF conventions.
- **Smart file listing**: Auto-detects .gguf files via HF API (`/api/models/{repo}/tree/main`), picks the first alphabetically.
- **Verified with real model**: Successfully downloaded `tinyllama-1.1b-chat-v1.0.Q2_K.gguf` (483MB) from `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`.
- **Clippy clean**: Fixed all warnings (collapsible if, unwrap_or_default, unused variable prefix with `_`).
- **opencode.json**: Updated permissions to `"ask"` for general bash, allowlisted cargo/pnpm/rustup/mise/mkdir/rm, specific git commands, denied git commit/push and gh actions.

### Known limitations

- Tensor type 14 (Q2_K) in GGUF not yet supported by `rusticide-model` -- need to extend tensor type enum.
- Sha256 verification skipped (would need HF API calls + auth).
- `--no-symlinks` flag accepted but not yet wired to logic.

### URLs

- HF model download: `https://huggingface.co/{repo}/resolve/main/{filename}`
- HF file list API: `https://huggingface.co/api/models/{repo}/tree/main`

## Session context (Jul 15, 2026)

### What was done

- **License correction**: Code files are `MIT OR Apache-2.0` (dual-license for patent protection),
  documentation files are `CC-BY-3.0+` (CII badge requirement, OSI-approved doc license).
- **Created license files**: `LICENSE` (MIT text), `LICENSE-APACHE` (Apache-2.0 text),
  `CC-BY-3.0.txt` (Creative Commons Attribution 3.0 Unported).
- **REUSE.toml**: Updated with per-file annotations for license text files and DCO.txt.
- **Doc SPDX headers**: `CII_BEST_PRACTICES.md` and `CONTRIBUTING.md` use inline
  CC-BY-3.0+ SPDX-License-Identifier headers.
- **AGENTS.md**: Added full License model section with reasoning.

### Known limitations

- (none)

### Reasoning: `MIT OR Apache-2.0` for code + `CC-BY-3.0+` for docs

**Code** uses `MIT OR Apache-2.0` because:

- Apache-2.0 provides an explicit patent grant (Section 3) that MIT alone lacks.
  This protects downstream users from patent litigation by contributors -- critical
  for an inference engine that may implement patented techniques.
- Dual-licensing gives downstream users choice: MIT for maximum compatibility,
  Apache-2.0 for patent protection.
- This is the Rust ecosystem standard, used by the vast majority of crates on crates.io.

**Documentation** uses `CC-BY-3.0+` because:

- MIT and Apache-2.0 are **software licenses** for "source code," not creative works.
  Creative Commons is the standard for documentation.
- The OpenSSF CII Best Practices badge explicitly requires a documentation license
  for its "FLOSS license" criterion.
- CC-BY-3.0 is the simplest OSI-approved CC license: share/adapt with attribution.
  No copyleft (ShareAlike), no no-derivatives (ND) restrictions.
- The `+` suffix ["or any later version"] future-proofs -- downstream can use CC-BY-4.0+.
- CC-BY-4.0 would also be acceptable; 3.0+ was chosen because it matches SPDX
  short identifier convention used by REUSE, and CII docs specifically cite 3.0+.

### Files changed

| File | Change |
|------|--------|
| `AGENTS.md` | Added License model section + this session context |
| `LICENSE` | Created (MIT text) |
| `LICENSE-APACHE` | Created (Apache-2.0 text) |
| `CC-BY-3.0.txt` | Created (CC-BY-3.0 Unported text) |
| `REUSE.toml` | Added annotations for LICENSE, LICENSE-APACHE, CC-BY-3.0.txt, DCO.txt |
| `CII_BEST_PRACTICES.md` | Added SPDX header (CC-BY-3.0+) |
| `CONTRIBUTING.md` | Fixed SPDX header format (CC-BY-3.0+) |
| `.rs` files (14) | Removed duplicate SPDX line that had CC-BY-3.0+ alongside MIT OR Apache-2.0 |

## License model

### Code: `MIT OR Apache-2.0`

Dual-licensing provides:

- **MIT**: simple permissive license, widely compatible, lets anyone use the code with minimal restrictions.
- **Apache-2.0**: MIT's permissions PLUS an explicit patent grant. Apache-2.0 Section 3 grants a patent license from contributors, which protects downstream users from patent litigation. This is the Rust ecosystem standard (`Cargo.toml` convention), followed by most crates published to crates.io.

Both licenses are OSI-approved. Users may choose either. `LICENSE` and `LICENSE-APACHE` contain the full texts.

### Docs: `CC-BY-3.0+`

Documentation files (`CONTRIBUTING.md`, `CII_BEST_PRACTICES.md`, etc.) use Creative Commons Attribution 3.0 Unported or any later version. Rationale:

- **Why not MIT/Apache-2.0 for docs?** MIT and Apache-2.0 are software licenses -- they cover "source code," not "creative works." While technically usable for docs, Creative Commons is the standard for documentation, and the OpenSSF CII Best Practices badge explicitly requires a **documentation license** for its "FLOSS license" criterion.
- **Why CC-BY-3.0?** It is the OSI-approved Creative Commons license with the simplest requirements: anyone can share/adapt the docs as long as they give attribution. No ShareAlike copyleft (which would prevent commercial reuse), no ND (which would prevent modification).
- **Why `+` (or any later version)?** Future-proofing -- downstream users can use CC-BY-4.0 or later if they prefer. The CII badge specifically recommends CC-BY-3.0+ or CC-BY-4.0.
- **What about CC-BY-4.0?** Also acceptable. 3.0+ was chosen to match the SPDX short identifier convention used by REUSE, and 3.0 is the version most commonly cited in badge documentation.

### Enforcement

- All files carry SPDX headers (inline or via `REUSE.toml` aggregate annotations).
- `reuse lint` in CI enforces compliance -- zero warnings required.
- Code files inherit `MIT OR Apache-2.0` from the `REUSE.toml` aggregate annotation.
- Doc files use inline `CC-BY-3.0+` SPDX license identifiers.
- License text files (`LICENSE`, `LICENSE-APACHE`, `CC-BY-3.0.txt`, `DCO.txt`) use explicit `REUSE.toml` annotations.
