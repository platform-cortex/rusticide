#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Render conventional-commit tooling configs from a single source of truth.

`commit-types.toml` is the only file authored by hand. This script renders the
derived regions of the consumer configs from it:

    committed.toml   whole file   (commit-message linter rules + allowed_types)
    cliff.toml       commit_parsers array, between sentinel markers
    .gitmessage      the type list, between sentinel markers

Usage:
    commit-config.py generate   # write the derived files
    commit-config.py check      # exit 1 (with a diff) if any derived file is stale

Zero third-party dependencies: reads TOML via the stdlib `tomllib` (Python 3.11+).
"""

from __future__ import annotations

import difflib
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "commit-types.toml"

# Sentinel markers delimiting the generated regions inside hand-maintained files.
# The marker lines themselves are preserved; only the content between them changes.
CLIFF_START = "# >>> commit-config:parsers (generated from commit-types.toml -- do not edit) >>>"
CLIFF_END = "# <<< commit-config:parsers <<<"
GITMSG_START = "# >>> commit-config:types (generated from commit-types.toml -- do not edit) >>>"
GITMSG_END = "# <<< commit-config:types <<<"

# Emitted verbatim at the head of every fully-generated file. Every line carries
# its own `#`, because a banner that relies on the caller to comment it silently
# produces invalid TOML the moment it grows past one line -- which is exactly how
# an SPDX-annotation pass stamping a header in here once broke the generator.
GEN_BANNER = """\
# SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# @generated from commit-types.toml by scripts/commit-config.py -- DO NOT EDIT"""


def toml_escape(value: str) -> str:
    """Escape a string for a TOML basic (double-quoted) string."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def load_manifest() -> dict:
    with MANIFEST.open("rb") as fh:
        return tomllib.load(fh)


# --------------------------------------------------------------------------- #
# Renderers -- each returns the exact text its target file should contain.
# --------------------------------------------------------------------------- #
def render_committed(manifest: dict) -> str:
    """Render the entire committed.toml file."""
    commit = manifest["commit"]
    allowed = [t["name"] for t in manifest["type"] if t.get("committed", True)]

    lines = [
        GEN_BANNER,
        "# Regenerate with `mise run commit-config`; `mise run commit-config:check` guards drift.",
        "#",
        "# committed ~ https://github.com/crate-ci/committed",
        f'style = "{commit["style"]}"',
        f'subject_length = {commit["subject_length"]}',
        f'subject_capitalized = {str(commit["subject_capitalized"]).lower()}',
        f'subject_not_punctuated = {str(commit["subject_not_punctuated"]).lower()}',
        f'imperative_subject = {str(commit["imperative_subject"]).lower()}',
        f'line_length = {commit["line_length"]}',
        f'no_wip = {str(commit["no_wip"]).lower()}',
        f'no_fixup = {str(commit["no_fixup"]).lower()}',
        f'merge_commit = {str(commit["merge_commit"]).lower()}',
        "allowed_types = [",
    ]
    lines += [f'  "{toml_escape(name)}",' for name in allowed]
    lines.append("]")
    return "\n".join(lines) + "\n"


def _ordered_parsers(manifest: dict) -> list[dict]:
    """Build the cliff parser list: skips first, then types+extras sorted by order."""
    parsers: list[dict] = []
    # 1. Exclusionary skip guards -- must precede the grouping parsers.
    for pat in manifest["cliff"]["skip_messages"]:
        parsers.append({"keys": [("message", pat)], "skip": True})

    # 2. Per-type and extra parsers, stable-sorted by `order` (manifest order breaks ties).
    grouped: list[tuple[int, int, dict]] = []
    seq = 0
    for t in manifest["type"]:
        pattern = t.get("pattern", f"^{t['name']}")
        group = f"<!-- {t['order']} -->{t['emoji']} {t['group']}"
        grouped.append((t["order"], seq, {"keys": [("message", pattern)], "group": group}))
        seq += 1
    for e in manifest["cliff"].get("extra_parser", []):
        group = f"<!-- {e['order']} -->{e['emoji']} {e['group']}"
        grouped.append((e["order"], seq, {"keys": [(e["field"], e["pattern"])], "group": group}))
        seq += 1

    grouped.sort(key=lambda item: (item[0], item[1]))
    parsers += [p for _, _, p in grouped]
    return parsers


def render_cliff_block(manifest: dict) -> str:
    """Render the `commit_parsers = [...]` assignment (the region between markers)."""
    lines = ["commit_parsers = ["]
    for p in _ordered_parsers(manifest):
        fields = ", ".join(f'{k} = "{toml_escape(v)}"' for k, v in p["keys"])
        if p.get("skip"):
            lines.append(f"  {{ {fields}, skip = true }},")
        else:
            lines.append(f'  {{ {fields}, group = "{toml_escape(p["group"])}" }},')
    lines.append("]")
    return "\n".join(lines)


def render_gitmessage_block(manifest: dict) -> str:
    """Render the commented type list shown in the git editor."""
    width = max(len(t["name"]) for t in manifest["type"])
    lines = ["# Type can be:"]
    for t in manifest["type"]:
        lines.append(f"#   {t['name']:<{width}}  ({t['desc']})")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Splicing -- replace the content between two marker lines, keeping the markers.
# --------------------------------------------------------------------------- #
def splice(text: str, start: str, end: str, inner: str) -> str:
    lines = text.splitlines()
    try:
        i = lines.index(start)
        j = lines.index(end)
    except ValueError as exc:
        raise SystemExit(
            f"error: markers {start!r}/{end!r} not found -- add them to the target file first"
        ) from exc
    if j <= i:
        raise SystemExit(f"error: end marker precedes start marker ({start!r})")
    new_lines = lines[: i + 1] + inner.splitlines() + lines[j:]
    trailing = "\n" if text.endswith("\n") else ""
    return "\n".join(new_lines) + trailing


def desired_outputs(manifest: dict) -> dict[Path, str]:
    """Map each target path to the full text it should have."""
    cliff_path = ROOT / "cliff.toml"
    gitmsg_path = ROOT / ".gitmessage"
    return {
        ROOT / "committed.toml": render_committed(manifest),
        cliff_path: splice(
            cliff_path.read_text(), CLIFF_START, CLIFF_END, render_cliff_block(manifest)
        ),
        gitmsg_path: splice(
            gitmsg_path.read_text(), GITMSG_START, GITMSG_END, render_gitmessage_block(manifest)
        ),
    }


def _taplo_format(path: Path) -> None:
    """Format `path` in place with taplo, surfacing taplo's own diagnostics on failure.

    `check=True` alone raises a CalledProcessError whose message is a bare exit
    status, discarding the parse error taplo wrote to stderr -- the one thing that
    explains what actually went wrong.
    """
    proc = subprocess.run(
        ["taplo", "format", str(path)], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "(no output)"
        raise SystemExit(f"error: taplo could not format {path}:\n{detail}")


def _normalized(text: str) -> str:
    """Return `text` as taplo would format it, without touching any tracked file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        scratch = Path(tmpdir) / "normalize.toml"
        scratch.write_text(text)
        _taplo_format(scratch)
        return scratch.read_text()


def _format_generated() -> None:
    """Normalize generated files through taplo to match repo formatting rules."""
    for f in (ROOT / "committed.toml", ROOT / "cliff.toml"):
        _taplo_format(f)


def cmd_generate() -> int:
    for path, content in desired_outputs(load_manifest()).items():
        path.write_text(content)
        print(f"wrote {path.relative_to(ROOT)}")
    _format_generated()
    return 0


def cmd_check() -> int:
    manifest = load_manifest()
    drifted = False
    for path, want in desired_outputs(manifest).items():
        have = path.read_text() if path.exists() else ""
        if have != want:
            # The on-disk file has been through taplo (key reordering, spacing), so a
            # raw mismatch is not drift yet -- normalize both sides and compare again.
            # Both are normalized as scratch copies: `check` must never write to the
            # tree it audits, or it would quietly repair the drift it exists to report.
            # .gitmessage is not TOML, so it is compared as authored.
            if path.suffix == ".toml":
                want_norm = _normalized(want)
                have_norm = _normalized(have)
            else:
                want_norm, have_norm = want, have
            if have_norm != want_norm:
                drifted = True
                rel = path.relative_to(ROOT)
                print(f"drift: {rel} is stale -- run `mise run commit-config`", file=sys.stderr)
                diff = difflib.unified_diff(
                    have_norm.splitlines(keepends=True),
                    want_norm.splitlines(keepends=True),
                    fromfile=f"{rel} (on disk)",
                    tofile=f"{rel} (expected)",
                )
                sys.stderr.writelines(diff)
    return 1 if drifted else 0


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "generate"
    if cmd == "generate":
        return cmd_generate()
    if cmd == "check":
        return cmd_check()
    print(f"usage: {Path(argv[0]).name} [generate|check]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
