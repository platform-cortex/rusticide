<!--
SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Tokenizer performance milestone (notes and plan)

Status: done on branch `perf/tokenizer`, 2026-10-31. Scope is
performance only: the encode path of `rusticide-tokenizer`, measured against
HF `tokenizers`, llama.cpp and OpenAI's tiktoken, with token ids that stay
identical to llama.cpp (the project's reference).

## Benchmark

A one-off harness outside the repository encodes three corpora with each
tokenizer, single-threaded, release build, median of 7 runs (3 for the
pathological case), on the Apple M1 Pro (8 cores, 32 GiB, macOS arm64):

- docs: every `.md` file of the repository, 306373 bytes of English prose;
- code: every `.rs` file of `crates/`, 605994 bytes;
- spaces: one run of 200000 spaces, a single pre-tokenized piece.

Real text, not repeated samples, so caches keyed on words do not get an unfair
advantage. rusticide and HF use the Qwen2.5 0.5B vocabulary (GGUF and
`tokenizer.json`); llama.cpp is `llama-tokenize --no-escape` on the same GGUF
(process time minus a one-token baseline, so it also pays for printing ids);
tiktoken-rs 0.12.1 uses `o200k_base`, a different vocabulary and pattern, so
it is a speed reference only.

Throughput in MB/s:

- rusticide at v0.4.1 (start): docs 5.44, code 5.56, spaces 82 ms.
- rusticide on `perf/tokenizer`: docs 24.4, code 28.3 to 29.2, spaces 68 ms.
- HF `tokenizers` 0.23.2 (fancy-regex build): docs 5.2, code 5.0, spaces
  63 ms.
- llama.cpp `llama-tokenize` 0.8.0: docs 3.2, code 3.4, spaces 105 ms.
- tiktoken-rs `o200k_base`: docs 23.6 to 25.6, code 23.8 to 24.6, spaces
  42 ms.

Ids: rusticide equals llama.cpp on all three corpora. It equals HF everywhere
except where the text contains a literal `</s>`, which llama.cpp (and now
rusticide) promote to a special token and HF does not.

## What the profiles showed

Sampled with macOS `sample` and rendered with `inferno` (`cargo flamegraph`
0.6.14 records with `xctrace`, but cannot collapse this Xcode's trace format).

1. 32 percent: compiling the pre-tokenizer patterns on every encode. Fixed by
   compiling each pattern set once per process (`OnceLock`).
2. 36 percent: the BPE merge for every piece, even repeated words. Fixed by
   reusing the ids of a piece already seen in the same encode.
3. 42 percent: fancy-regex's backtracking VM, entered only because of the one
   look-ahead `\s+(?!\S)`. Fixed by matching `\s+` on `regex-automata`'s
   lazy DFA and replaying the look-ahead in code (a space run followed by a
   non-space character gives its last character back). An equivalence test
   checks every pre-tokenizer against the verbatim llama.cpp patterns on
   fancy-regex over 4000 generated texts.
4. 27 percent: heap push/pop for short words. Fixed by merging words of at
   most 16 symbols with a linear scan over cached pair ranks; longer pieces
   keep the O(n log n) heap.
5. 7 percent: SipHash in the merge table. Fixed with a multiply-rotate hasher
   for that table only (its keys come from the vocabulary, never from the
   text); the piece cache, keyed on text, keeps SipHash.
6. The 200000-space piece then took 68 ms (tiktoken 42 ms) on the heap path,
   whose entries copied the whole pair. Fixed by queueing only
   `(rank, position)` and treating an entry as current while that position
   still holds that rank, as tiktoken's `_byte_pair_merge_large` does: 36 ms.

## Supporting tiktoken vocabularies

Goal: run `cl100k_base` and `o200k_base` inside rusticide, so the comparison
with tiktoken is like for like (same vocabulary, same ids), and beat it.

What tiktoken is, from tiktoken-rs 0.12.1 (`src/tiktoken_ext/openai_public.rs`,
`src/vendor_tiktoken.rs`):

- Vocabulary: a `.tiktoken` file, one `base64(token bytes) rank` line per
  token; the rank is the token id. 100256 lines for `cl100k_base`, 199998 for
  `o200k_base`. Tokens are raw bytes: there is no GPT-2 byte-to-unicode
  mapping.
- Special tokens: a fixed table per encoding, not in the file (for example
  `<|endoftext|>` is 100257 in `cl100k_base` and 199999 in `o200k_base`).
- Merge rule: repeatedly merge the adjacent pair whose concatenated bytes have
  the lowest rank, leftmost on ties; pieces under a size threshold use a
  linear scan, longer ones a heap. Before merging, a piece that is itself a
  token is emitted directly.
- Pre-tokenizer: one regex per encoding. `cl100k_base` uses possessive
  quantifiers (`?+`, `++`, `{1,3}+`, `*+`) and `\s++$`; `o200k_base` uses
  case-split letter classes and the same `\s+(?!\S)` look-ahead.

Implemented (`Tokenizer::from_tiktoken`, `TiktokenEncoding`), reusing what
exists:

1. Loader: `Tokenizer::from_tiktoken(file bytes, encoding)` decodes the
   base64 lines (a small in-crate decoder, no new dependency) into the
   vocabulary, with ids equal to ranks, plus the encoding's special-token
   table.
2. Merge table: derive the existing `MergeLookup` from ranks. For every token
   `T` of two or more bytes and every split `T = A + B` where `A` and `B` are
   tokens, insert `(A, B) -> (rank(T), T)`. The current engine then produces
   exactly tiktoken's merges, because both merge the lowest-ranked concatenation
   first and break ties leftmost.
3. Raw bytes: a byte-level mode without the GPT-2 remapping, so the 256 byte
   tokens are looked up by their raw byte and decode concatenates raw bytes.
4. Patterns: add both regexes. Possessive quantifiers become greedy ones (they
   only stop backtracking, and the DFA never backtracks); `\s++$` and
   `\s*[\r\n]` are whitespace-only alternatives that come before the
   look-ahead, so the look-ahead replay must skip matches that end in a line
   break or at the end of the piece. The equivalence test gains the two
   patterns, checked against fancy-regex running the originals.
5. Whole-piece lookup: emit a piece that is itself a token without merging.
   For rank-trained vocabularies this is exact by construction (tiktoken relies
   on it). For merge-list vocabularies like Qwen's it is not guaranteed, so it
   stays behind a flag set only by the tiktoken loader.
6. Verification: ids equal to tiktoken-rs on the three corpora for both
   encodings, then the head-to-head benchmark on the same vocabulary.

Why it also matters beyond the benchmark: milestone 3's BitNet model uses the
Llama 3 vocabulary, which is tiktoken-derived; points 2, 4 and 5 are the same
machinery that path will lean on.

## Same-vocabulary result

`Tokenizer::from_tiktoken` loaded with the `.tiktoken` files tiktoken-rs
0.12.1 embeds, against tiktoken-rs `encode_with_special_tokens` on the same
corpora and machine (rusticide load: 49 ms for `cl100k_base`, 102 ms for
`o200k_base`). Ids are identical in all six cases; speed-up over tiktoken:

- `cl100k_base`: docs 2.23x (35.7 against 16.0 MB/s), code 2.57x (37.0
  against 14.4), spaces 1.08x (36.7 against 39.6 ms).
- `o200k_base`: docs 1.47x (34.1 against 23.3 MB/s), code 1.69x (39.6
  against 23.5), spaces 1.16x (36.1 against 42.1 ms).
