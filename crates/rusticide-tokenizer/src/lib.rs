// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;

use rusticide_model::{GGUFValue, Model};

// ---------------------------------------------------------------------------
// TokenizerModel
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerModel {
    LLaMA,
    GPT2,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("token {0} out of range (vocab size: {1})")]
    TokenOutOfRange(u32, usize),
    #[error("unknown model type: {0}")]
    UnknownModel(String),
    #[error("missing required tokenizer metadata: {0}")]
    MissingMetadata(&'static str),
    #[error("cannot encode: {0}")]
    EncodeError(String),
    #[error("cannot decode: {0}")]
    DecodeError(String),
    #[error("invalid token ID (negative): {0}")]
    InvalidId(i32),
    #[error("vocabulary size exceeds u32 range")]
    VocabOverflow,
}

pub type Result<T> = std::result::Result<T, TokenizerError>;

// ---------------------------------------------------------------------------
// GGUFValueExt — extension trait because GGUFValue is from another crate
// ---------------------------------------------------------------------------

trait GGUFValueExt {
    fn as_string(&self) -> Result<String>;
    fn as_array(&self) -> Result<&Vec<GGUFValue>>;
    fn as_f32(&self) -> Result<f32>;
    fn as_bool(&self) -> Result<bool>;
    fn as_u32_lossy(&self) -> Result<u32>;
}

impl GGUFValueExt for GGUFValue {
    fn as_string(&self) -> Result<String> {
        match self {
            Self::String(s) => Ok(s.clone()),
            _ => Err(TokenizerError::MissingMetadata(
                "expected string value type",
            )),
        }
    }

    fn as_array(&self) -> Result<&Vec<GGUFValue>> {
        match self {
            Self::Array(arr) => Ok(arr),
            _ => Err(TokenizerError::MissingMetadata("expected array value type")),
        }
    }

    fn as_f32(&self) -> Result<f32> {
        match self {
            Self::Float32(f) => Ok(*f),
            _ => Err(TokenizerError::MissingMetadata(
                "expected float32 value type",
            )),
        }
    }

    fn as_bool(&self) -> Result<bool> {
        match self {
            Self::Bool(b) => Ok(*b),
            _ => Err(TokenizerError::MissingMetadata("expected bool value type")),
        }
    }

    fn as_u32_lossy(&self) -> Result<u32> {
        match self {
            Self::Uint32(u) => Ok(*u),
            Self::Int32(i) => u32::try_from(*i).map_err(|_| TokenizerError::InvalidId(*i)),
            Self::Uint64(u) => {
                u32::try_from(*u).map_err(|_| TokenizerError::MissingMetadata("id exceeds u32"))
            }
            Self::Int64(i) => {
                u32::try_from(*i).map_err(|_| TokenizerError::MissingMetadata("id exceeds u32"))
            }
            _ => Err(TokenizerError::MissingMetadata(
                "expected integer id value type",
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Trie
// ---------------------------------------------------------------------------

struct TrieNode {
    children: Vec<(u8, usize)>,
    token_id: Option<u32>,
}

struct Trie {
    nodes: Vec<TrieNode>,
}

impl Trie {
    fn new() -> Self {
        Self {
            nodes: vec![TrieNode {
                children: Vec::new(),
                token_id: None,
            }],
        }
    }

    fn insert(&mut self, bytes: &[u8], token_id: u32) {
        let mut idx = 0usize;
        for &b in bytes {
            let pos = self.nodes[idx].children.iter().position(|(c, _)| *c == b);
            if let Some(p) = pos {
                idx = self.nodes[idx].children[p].1;
            } else {
                let new_idx = self.nodes.len();
                self.nodes.push(TrieNode {
                    children: Vec::new(),
                    token_id: None,
                });
                self.nodes[idx].children.push((b, new_idx));
                idx = new_idx;
            }
        }
        self.nodes[idx].token_id = Some(token_id);
    }

    /// Returns all (`token_id`, `byte_length`, score) that match at `start`.
    fn prefixes(&self, bytes: &[u8], start: usize, scores: &[f32]) -> Vec<(u32, usize, f32)> {
        let mut result = Vec::new();
        let mut idx = 0usize;
        let mut len = 0usize;
        for &b in &bytes[start..] {
            let pos = self.nodes[idx].children.iter().position(|(c, _)| *c == b);
            if let Some(p) = pos {
                idx = self.nodes[idx].children[p].1;
                len += 1;
                if let Some(tid) = self.nodes[idx].token_id {
                    result.push((tid, len, scores[tid as usize]));
                }
            } else {
                break;
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct Tokenizer {
    model: TokenizerModel,
    vocab: Vec<Vec<u8>>,
    scores: Vec<f32>,
    bos_id: u32,
    eos_id: u32,
    add_bos: bool,
    add_space_prefix: bool,
    merge_info: HashMap<(u32, u32), (usize, u32)>,
}

impl Tokenizer {
    /// Construct a tokenizer directly from values (useful for testing).
    #[must_use]
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(
        model: TokenizerModel,
        vocab: Vec<&str>,
        scores: Option<Vec<f32>>,
        bos_id: u32,
        eos_id: u32,
        add_bos: bool,
    ) -> Self {
        let vocab_bytes: Vec<Vec<u8>> = vocab.iter().map(|s| s.as_bytes().to_vec()).collect();
        let scores_vec = scores.unwrap_or_else(|| vec![0.0; vocab_bytes.len()]);
        // add_space_prefix isn't a `new()` parameter (public API, must stay
        // stable) -- derive it from the model type instead, matching
        // llama.cpp's default (true for LLaMA/SPM, false for GPT-2/BPE).
        let add_space_prefix = model == TokenizerModel::LLaMA;
        Self {
            model,
            vocab: vocab_bytes,
            scores: scores_vec,
            bos_id,
            eos_id,
            add_bos,
            add_space_prefix,
            merge_info: HashMap::new(),
        }
    }

    /// Construct a tokenizer from GGUF model metadata.
    ///
    /// Reads `tokenizer.ggml.*` keys from the model's metadata store.
    ///
    /// # Errors
    ///
    /// Returns `MissingMetadata` if required keys are missing,
    /// `UnknownModel` if the model type is not recognized.
    pub fn from_gguf(model: &Model) -> Result<Self> {
        let model_str = model
            .metadata_value("tokenizer.ggml.model")
            .ok_or(TokenizerError::MissingMetadata("tokenizer.ggml.model"))?;
        let model_str = model_str.as_string()?;
        let tokenizer_model = match model_str.as_str() {
            "llama" => TokenizerModel::LLaMA,
            "gpt2" => TokenizerModel::GPT2,
            other => return Err(TokenizerError::UnknownModel(other.to_string())),
        };

        // Tokens
        let tokens_val = model
            .metadata_value("tokenizer.ggml.tokens")
            .ok_or(TokenizerError::MissingMetadata("tokenizer.ggml.tokens"))?;
        let token_arr = tokens_val.as_array()?;
        let mut vocab = Vec::with_capacity(token_arr.len());
        let mut vocab_strings = Vec::with_capacity(token_arr.len());
        for val in token_arr {
            let s = val.as_string()?;
            vocab_strings.push(s.clone());
            vocab.push(s.into_bytes());
        }

        // Scores (optional)
        let scores: Vec<f32> = match model.metadata_value("tokenizer.ggml.scores") {
            Some(v) => {
                let arr = v.as_array()?;
                arr.iter().map(|v| v.as_f32().unwrap_or(0.0)).collect()
            }
            None => vec![0.0; vocab.len()],
        };

        // BOS / EOS ids. Real llama.cpp GGUFs export these as UINT32 under
        // the *_token_id keys; treat them as optional with llama defaults.
        let bos_default = u32::from(tokenizer_model == TokenizerModel::LLaMA);
        let bos_id = match model.metadata_value("tokenizer.ggml.bos_token_id") {
            Some(v) => v.as_u32_lossy()?,
            None => bos_default,
        };
        let eos_id = match model.metadata_value("tokenizer.ggml.eos_token_id") {
            Some(v) => v.as_u32_lossy()?,
            None => 2,
        };

        // add_bos — default true for LLaMA, false for GPT2
        let add_bos_default = tokenizer_model == TokenizerModel::LLaMA;
        let add_bos = model
            .metadata_value("tokenizer.ggml.add_bos_token")
            .map_or(add_bos_default, |v| v.as_bool().unwrap_or(add_bos_default));

        // add_space_prefix -- default true for LLaMA, false for GPT2. This
        // controls whether encode() prepends a dummy leading space before
        // SPM space normalization, mirroring llama.cpp's own default.
        let add_space_prefix_default = tokenizer_model == TokenizerModel::LLaMA;
        let add_space_prefix = model
            .metadata_value("tokenizer.ggml.add_space_prefix")
            .map_or(add_space_prefix_default, |v| {
                v.as_bool().unwrap_or(add_space_prefix_default)
            });

        // Merge table (GPT-2 only, optional)
        let merge_info = if tokenizer_model == TokenizerModel::GPT2 {
            let token_to_id: HashMap<String, u32> = vocab_strings
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    let id = u32::try_from(i).map_err(|_| TokenizerError::VocabOverflow)?;
                    Ok((s.clone(), id))
                })
                .collect::<Result<_>>()?;
            if model.metadata_value("tokenizer.ggml.merges").is_some() {
                Self::build_merge_lookup(&token_to_id, model)?
            } else {
                HashMap::new()
            }
        } else {
            HashMap::new()
        };

        Ok(Self {
            model: tokenizer_model,
            vocab,
            scores,
            bos_id,
            eos_id,
            add_bos,
            add_space_prefix,
            merge_info,
        })
    }

    /// Parse BPE merge entries from GGUF metadata.
    fn build_merge_lookup(
        token_to_id: &HashMap<String, u32>,
        model: &Model,
    ) -> Result<HashMap<(u32, u32), (usize, u32)>> {
        let merges_val = model
            .metadata_value("tokenizer.ggml.merges")
            .ok_or(TokenizerError::MissingMetadata("tokenizer.ggml.merges"))?;
        let merges_arr = merges_val.as_array()?;
        let mut merge_info = HashMap::new();
        for (rank, merge_val) in merges_arr.iter().enumerate() {
            let pair_str = merge_val.as_string()?;
            let Some((left_str, right_str)) = pair_str.split_once(' ') else {
                continue;
            };
            let combined_str = format!("{left_str}{right_str}");
            // Look up the merged token in the vocabulary.  It must exist because
            // every merge pair produces a token that is stored in the vocab.
            let Some(&merged_id) = token_to_id.get(&combined_str) else {
                // If the merged string isn't found, skip — some vocabularies don't
                // store all merge results.
                continue;
            };
            // Look up left/right by their string representation.
            let Some(&left_id) = token_to_id.get(left_str) else {
                continue;
            };
            let Some(&right_id) = token_to_id.get(right_str) else {
                continue;
            };
            merge_info.insert((left_id, right_id), (rank, merged_id));
        }
        Ok(merge_info)
    }

    // -----------------------------------------------------------------------
    // Public accessors
    // -----------------------------------------------------------------------

    #[must_use]
    pub const fn model_type(&self) -> TokenizerModel {
        self.model
    }

    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    #[must_use]
    pub const fn bos_id(&self) -> u32 {
        self.bos_id
    }

    #[must_use]
    pub const fn eos_id(&self) -> u32 {
        self.eos_id
    }

    // -----------------------------------------------------------------------
    // Encoding
    // -----------------------------------------------------------------------

    /// Encode text to token IDs.
    ///
    /// # Errors
    ///
    /// Returns `EncodeError` if the text cannot be encoded.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        if text.is_empty() {
            return if self.add_bos {
                Ok(vec![self.bos_id])
            } else {
                Ok(Vec::new())
            };
        }

        let tokens = match self.model {
            TokenizerModel::LLaMA => self.encode_llama(text)?,
            TokenizerModel::GPT2 => self.encode_gpt2(text),
        };

        if self.add_bos {
            let mut result = vec![self.bos_id];
            result.extend(tokens);
            Ok(result)
        } else {
            Ok(tokens)
        }
    }

    /// `LLaMA` / `SentencePiece` unigram encoding via Viterbi DP.
    fn encode_llama(&self, text: &str) -> Result<Vec<u32>> {
        // llama.cpp SPM preprocessing: optionally prepend a dummy leading
        // space, then replace every ASCII space with the SentencePiece
        // word-initial marker (U+2581) that real llama vocabularies use for
        // word-initial pieces. This must happen before Viterbi, or
        // word-initial pieces (e.g. "<U+2581>hello") never match.
        let mut prefixed = String::with_capacity(text.len() + 1);
        if self.add_space_prefix {
            prefixed.push(' ');
        }
        prefixed.push_str(text);
        let normalized = prefixed.replace(' ', "\u{2581}");

        let bytes = normalized.as_bytes();
        let n = bytes.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        let byte_fallback = self.build_byte_fallback();
        let trie = self.build_trie();

        // back[pos] = (token_id, span_in_bytes) — span is the number of input
        // bytes this token covers.  For byte-fallback tokens the span is
        // always 1, not the vocab entry's byte length.
        let inf = f32::INFINITY;
        let mut dp = vec![inf; n + 1];
        let mut back: Vec<(u32, usize)> = vec![(0, 0); n + 1];
        dp[0] = 0.0;

        for i in 0..n {
            if !dp[i].is_finite() {
                // Position unreachable — try byte fallback
                if let Some(bf_id) = byte_fallback[bytes[i] as usize] {
                    let cost = dp[i] + (-self.scores[bf_id as usize]);
                    if cost < dp[i + 1] {
                        dp[i + 1] = cost;
                        back[i + 1] = (bf_id, 1);
                    }
                }
                continue;
            }

            let matches = trie.prefixes(bytes, i, &self.scores);
            if matches.is_empty() {
                // No vocabulary token matches at this position — use byte fallback
                if let Some(bf_id) = byte_fallback[bytes[i] as usize] {
                    let cost = dp[i] + (-self.scores[bf_id as usize]);
                    if cost < dp[i + 1] {
                        dp[i + 1] = cost;
                        back[i + 1] = (bf_id, 1);
                    }
                }
            } else {
                for (tid, tlen, score) in matches {
                    let cost = dp[i] + (-score);
                    let next = i + tlen;
                    if cost < dp[next] {
                        dp[next] = cost;
                        back[next] = (tid, tlen);
                    }
                }
            }
        }

        if !dp[n].is_finite() {
            return Err(TokenizerError::EncodeError(
                "Viterbi could not find a valid tokenization path".to_string(),
            ));
        }

        // Backtrack
        let mut result = Vec::new();
        let mut pos = n;
        while pos > 0 {
            let (tid, span) = back[pos];
            result.push(tid);
            pos -= span;
        }
        result.reverse();
        Ok(result)
    }

    /// GPT-2 byte-level BPE encoding.
    fn encode_gpt2(&self, text: &str) -> Vec<u32> {
        let words = Self::gpt2_pre_tokenize(text);
        let mut all_ids = Vec::new();

        for word in words {
            let word_ids = self.bpe_encode_word(&word);
            all_ids.extend(word_ids);
        }

        all_ids
    }

    /// Simple whitespace-based pre-tokenization for GPT-2.
    fn gpt2_pre_tokenize(text: &str) -> Vec<String> {
        let mut words = Vec::new();
        let mut current = String::new();
        for ch in text.chars() {
            if ch.is_whitespace() {
                if !current.is_empty() {
                    words.push(current);
                    current = String::new();
                }
            } else {
                current.push(ch);
            }
        }
        if !current.is_empty() {
            words.push(current);
        }
        words
    }

    /// Apply BPE merges to a single word — start from byte-level tokens.
    fn bpe_encode_word(&self, word: &str) -> Vec<u32> {
        let bytes = word.as_bytes();
        if bytes.is_empty() {
            return Vec::new();
        }
        // Start with byte tokens (byte value → token ID)
        let mut ids: Vec<u32> = bytes.iter().map(|&b| u32::from(b)).collect();

        loop {
            let mut best_rank = usize::MAX;
            let mut best_i = None;
            for i in 0..ids.len().saturating_sub(1) {
                if let Some(&(rank, _)) = self.merge_info.get(&(ids[i], ids[i + 1])) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_i = Some(i);
                    }
                }
            }
            let Some(i) = best_i else {
                break;
            };
            let merged_id = self.merge_info[&(ids[i], ids[i + 1])].1;
            ids[i] = merged_id;
            ids.remove(i + 1);
        }

        ids
    }

    // -----------------------------------------------------------------------
    // Decoding
    // -----------------------------------------------------------------------

    /// Decode token IDs back to text by joining their byte sequences.
    ///
    /// # Errors
    ///
    /// Returns `TokenOutOfRange` if any token ID exceeds the vocabulary.
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        let bytes = self.decode_to_bytes(tokens)?;
        Ok(String::from_utf8_lossy(&bytes).to_string())
    }

    /// Decode token IDs but filter special tokens (BOS/EOS) and join with
    /// appropriate spacing.
    ///
    /// # Errors
    ///
    /// Returns `TokenOutOfRange` if any token ID exceeds the vocabulary.
    pub fn decode_detokenized(&self, tokens: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();
        for &tid in tokens {
            if tid == self.bos_id || tid == self.eos_id {
                continue;
            }
            let idx = usize::try_from(tid)
                .map_err(|_| TokenizerError::TokenOutOfRange(tid, self.vocab.len()))?;
            if idx >= self.vocab.len() {
                return Err(TokenizerError::TokenOutOfRange(tid, self.vocab.len()));
            }
            if !bytes.is_empty() && self.add_space_before(tid) {
                bytes.push(b' ');
            }
            bytes.extend_from_slice(&self.vocab[idx]);
        }
        Ok(String::from_utf8_lossy(&bytes).to_string())
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Decode to raw bytes without UTF-8 conversion.
    fn decode_to_bytes(&self, tokens: &[u32]) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        for &tid in tokens {
            let idx = usize::try_from(tid)
                .map_err(|_| TokenizerError::TokenOutOfRange(tid, self.vocab.len()))?;
            if idx >= self.vocab.len() {
                return Err(TokenizerError::TokenOutOfRange(tid, self.vocab.len()));
            }
            bytes.extend_from_slice(&self.vocab[idx]);
        }
        Ok(bytes)
    }

    /// Heuristic: should we insert a space before this token during detokenization?
    fn add_space_before(&self, tid: u32) -> bool {
        let idx = tid as usize;
        if idx >= self.vocab.len() {
            return false;
        }
        let bytes = &self.vocab[idx];
        if bytes.is_empty() {
            return false;
        }
        // For LLaMA: tokens starting with a non-byte-fallback character don't
        // need a space before them if they are the first piece.
        // In SentencePiece, tokens that start with `▁` (U+2581) represent a
        // space-prefixed token.  We check if the first byte is 0xE2 0x96 0x81
        // (UTF-8 encoding of U+2581).
        let spm_marker = [0xE2, 0x96, 0x81];
        bytes.starts_with(&spm_marker)
    }

    /// Build a trie over all vocabulary entries.
    fn build_trie(&self) -> Trie {
        let mut trie = Trie::new();
        for (id, b) in self.vocab.iter().enumerate() {
            if !b.is_empty() {
                trie.insert(b, u32::try_from(id).unwrap_or(u32::MAX));
            }
        }
        trie
    }

    /// Build a byte-fallback lookup table from `<0xXX>` tokens.
    fn build_byte_fallback(&self) -> [Option<u32>; 256] {
        let mut fallback = [None; 256];
        for (id, bytes) in self.vocab.iter().enumerate() {
            if bytes.len() == 6
                && bytes[0] == b'<'
                && bytes[1] == b'0'
                && bytes[2] == b'x'
                && bytes[5] == b'>'
            {
                // Parse two hex digits
                let hi = hex_nibble(bytes[3]);
                let lo = hex_nibble(bytes[4]);
                if let (Some(h), Some(l)) = (hi, lo) {
                    let byte_val = (h << 4) | l;
                    fallback[byte_val as usize] = Some(u32::try_from(id).unwrap_or(u32::MAX));
                }
            }
        }
        fallback
    }
}

/// Convert an ASCII hex digit to its value.
#[must_use]
const fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    /// Write a GGUF string (length-prefixed UTF-8) into `buf`.
    fn gguf_write_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    /// Write a KV pair whose value is a String.
    fn gguf_write_kv_str(buf: &mut Vec<u8>, key: &str, val: &str) {
        gguf_write_string(buf, key);
        buf.extend_from_slice(&8u32.to_le_bytes()); // String type tag
        gguf_write_string(buf, val);
    }

    /// Write a KV pair whose value is an Int32.
    fn gguf_write_kv_i32(buf: &mut Vec<u8>, key: &str, val: i32) {
        gguf_write_string(buf, key);
        buf.extend_from_slice(&5u32.to_le_bytes()); // Int32 type tag
        buf.extend_from_slice(&val.to_le_bytes());
    }

    /// Write a KV pair whose value is a Uint32.
    fn gguf_write_kv_u32(buf: &mut Vec<u8>, key: &str, val: u32) {
        gguf_write_string(buf, key);
        buf.extend_from_slice(&4u32.to_le_bytes()); // Uint32 type tag
        buf.extend_from_slice(&val.to_le_bytes());
    }

    /// Write a KV pair whose value is a Bool.
    fn gguf_write_kv_bool(buf: &mut Vec<u8>, key: &str, val: bool) {
        gguf_write_string(buf, key);
        buf.extend_from_slice(&7u32.to_le_bytes()); // Bool type tag
        buf.push(u8::from(val));
    }

    /// Write a KV pair whose value is an Array-of-String.
    fn gguf_write_kv_str_array(buf: &mut Vec<u8>, key: &str, items: &[&str]) {
        gguf_write_string(buf, key);
        buf.extend_from_slice(&9u32.to_le_bytes()); // Array type tag
        buf.extend_from_slice(&8u32.to_le_bytes()); // item type = String
        buf.extend_from_slice(&(items.len() as u64).to_le_bytes());
        for item in items {
            gguf_write_string(buf, item);
        }
    }

    /// Write a KV pair whose value is an Array-of-Float32.
    fn gguf_write_kv_f32_array(buf: &mut Vec<u8>, key: &str, items: &[f32]) {
        gguf_write_string(buf, key);
        buf.extend_from_slice(&9u32.to_le_bytes()); // Array type tag
        buf.extend_from_slice(&6u32.to_le_bytes()); // item type = Float32
        buf.extend_from_slice(&(items.len() as u64).to_le_bytes());
        for &v in items {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }

    /// Build a miniature GGUF binary containing tokenizer metadata.
    ///
    /// `bos_id` / `eos_id` are `Option` so callers can omit the
    /// `tokenizer.ggml.{bos,eos}_token_id` keys entirely and exercise the
    /// default-value path in `Tokenizer::from_gguf`.
    fn build_tokenizer_gguf(
        model_type: &str,
        tokens: &[&str],
        scores: Option<&[f32]>,
        bos_id: Option<i32>,
        eos_id: Option<i32>,
        add_bos: Option<bool>,
        merges: Option<&[&str]>,
        add_space_prefix: Option<bool>,
    ) -> Vec<u8> {
        let mut buf = Vec::new();

        // --- Header ---
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        // KV count placeholder — patched at the end
        let kv_count_pos = buf.len();
        buf.extend_from_slice(&0u64.to_le_bytes());

        let mut kv_count = 0u64;

        // --- tokenizer.ggml.model (String) ---
        gguf_write_kv_str(&mut buf, "tokenizer.ggml.model", model_type);
        kv_count += 1;

        // --- tokenizer.ggml.tokens (Array[String]) ---
        gguf_write_kv_str_array(&mut buf, "tokenizer.ggml.tokens", tokens);
        kv_count += 1;

        // --- tokenizer.ggml.scores (Array[Float32], optional) ---
        if let Some(sc) = scores {
            gguf_write_kv_f32_array(&mut buf, "tokenizer.ggml.scores", sc);
            kv_count += 1;
        }

        // --- tokenizer.ggml.bos_token_id (Int32, optional) ---
        if let Some(bos) = bos_id {
            gguf_write_kv_i32(&mut buf, "tokenizer.ggml.bos_token_id", bos);
            kv_count += 1;
        }

        // --- tokenizer.ggml.eos_token_id (Int32, optional) ---
        if let Some(eos) = eos_id {
            gguf_write_kv_i32(&mut buf, "tokenizer.ggml.eos_token_id", eos);
            kv_count += 1;
        }

        // --- tokenizer.ggml.add_bos_token (Bool, optional) ---
        if let Some(ab) = add_bos {
            gguf_write_kv_bool(&mut buf, "tokenizer.ggml.add_bos_token", ab);
            kv_count += 1;
        }

        // --- tokenizer.ggml.merges (Array[String], optional) ---
        if let Some(mg) = merges {
            gguf_write_kv_str_array(&mut buf, "tokenizer.ggml.merges", mg);
            kv_count += 1;
        }

        // --- tokenizer.ggml.add_space_prefix (Bool, optional) ---
        if let Some(asp) = add_space_prefix {
            gguf_write_kv_bool(&mut buf, "tokenizer.ggml.add_space_prefix", asp);
            kv_count += 1;
        }

        // Patch KV count at the reserved position
        buf[kv_count_pos..kv_count_pos + 8].copy_from_slice(&kv_count.to_le_bytes());

        buf
    }

    fn write_temp_gguf(data: &[u8]) -> std::path::PathBuf {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "test_tok_{}_{}.gguf",
            std::process::id(),
            COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::write(&path, data).unwrap();
        path
    }

    /// Create a simple LLaMA-style tokenizer via GGUF.
    fn create_llama_tokenizer_gguf() -> Vec<u8> {
        build_tokenizer_gguf(
            "llama",
            &["<s>", "</s>", "<unk>", "hello", " world", "hell", "o", "lo"],
            Some(&[0.0, 0.0, 0.0, -2.0, -1.5, -1.0, -0.5, -0.8]),
            Some(0),
            Some(1),
            Some(true),
            None,
            None,
        )
    }

    /// Build a minimal llama tokenizer GGUF using the REAL key names that
    /// llama.cpp actually exports: `tokenizer.ggml.bos_token_id` /
    /// `eos_token_id`, both written as UINT32 (GGUF type 4).
    fn write_tokenizer_gguf_real_keys() -> std::path::PathBuf {
        let mut buf = Vec::new();

        // --- Header ---
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        let kv_count_pos = buf.len();
        buf.extend_from_slice(&0u64.to_le_bytes());

        let mut kv_count = 0u64;

        gguf_write_kv_str(&mut buf, "general.architecture", "llama");
        kv_count += 1;
        gguf_write_kv_str(&mut buf, "tokenizer.ggml.model", "llama");
        kv_count += 1;
        gguf_write_kv_str_array(&mut buf, "tokenizer.ggml.tokens", &["<s>", "</s>", "<unk>"]);
        kv_count += 1;
        gguf_write_kv_u32(&mut buf, "tokenizer.ggml.bos_token_id", 1);
        kv_count += 1;
        gguf_write_kv_u32(&mut buf, "tokenizer.ggml.eos_token_id", 2);
        kv_count += 1;

        buf[kv_count_pos..kv_count_pos + 8].copy_from_slice(&kv_count.to_le_bytes());

        write_temp_gguf(&buf)
    }

    // -----------------------------------------------------------------------
    // Tests: construction from GGUF
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_reads_real_bos_eos_token_id_keys() {
        // Minimal llama tokenizer GGUF using tokenizer.ggml.bos_token_id /
        // eos_token_id as UINT32 (type 4), 3 tokens, model = "llama".
        let path = write_tokenizer_gguf_real_keys();
        let model = Model::load(&path).unwrap();

        let tok = Tokenizer::from_gguf(&model).expect("from_gguf must accept real keys");
        assert_eq!(tok.bos_id(), 1);
        assert_eq!(tok.eos_id(), 2);

        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_from_gguf_llama() {
        let gguf = create_llama_tokenizer_gguf();
        let path = write_temp_gguf(&gguf);
        let model = Model::load(&path).unwrap();
        let tok = Tokenizer::from_gguf(&model).unwrap();

        assert_eq!(tok.model_type(), TokenizerModel::LLaMA);
        assert_eq!(tok.vocab_size(), 8);
        assert_eq!(tok.bos_id(), 0);
        assert_eq!(tok.eos_id(), 1);
        assert!(tok.add_bos);

        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_from_gguf_gpt2() {
        let gguf = build_tokenizer_gguf(
            "gpt2",
            &["!", "\"", "#", "$", "%"],
            None,
            Some(0),
            Some(1),
            Some(false),
            None,
            None,
        );
        let path = write_temp_gguf(&gguf);
        let model = Model::load(&path).unwrap();
        let tok = Tokenizer::from_gguf(&model).unwrap();

        assert_eq!(tok.model_type(), TokenizerModel::GPT2);
        assert_eq!(tok.vocab_size(), 5);
        assert!(!tok.add_bos);

        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_from_gguf_unknown_model() {
        let gguf = build_tokenizer_gguf(
            "unknown",
            &["a"],
            None,
            Some(0),
            Some(1),
            Some(false),
            None,
            None,
        );
        let path = write_temp_gguf(&gguf);
        let model = Model::load(&path).unwrap();
        let err = Tokenizer::from_gguf(&model).unwrap_err();
        assert!(matches!(err, TokenizerError::UnknownModel(_)));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_from_gguf_missing_required() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // 0 KV pairs
        let path = write_temp_gguf(&buf);
        let model = Model::load(&path).unwrap();
        let err = Tokenizer::from_gguf(&model).unwrap_err();
        assert!(matches!(err, TokenizerError::MissingMetadata(_)));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_from_gguf_add_bos_default_llama() {
        // No add_bos key — should default to true for LLaMA
        let gguf = build_tokenizer_gguf(
            "llama",
            &["a", "b"],
            None,
            Some(0),
            Some(1),
            None,
            None,
            None,
        );
        let path = write_temp_gguf(&gguf);
        let model = Model::load(&path).unwrap();
        let tok = Tokenizer::from_gguf(&model).unwrap();
        assert!(tok.add_bos);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_from_gguf_add_bos_default_gpt2() {
        // No add_bos key — should default to false for GPT2
        let gguf = build_tokenizer_gguf(
            "gpt2",
            &["a", "b"],
            None,
            Some(0),
            Some(1),
            None,
            None,
            None,
        );
        let path = write_temp_gguf(&gguf);
        let model = Model::load(&path).unwrap();
        let tok = Tokenizer::from_gguf(&model).unwrap();
        assert!(!tok.add_bos);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_from_gguf_bos_eos_default_llama() {
        // Neither tokenizer.ggml.bos_token_id nor eos_token_id is present --
        // LLaMA defaults are bos = 1, eos = 2.
        let gguf = build_tokenizer_gguf(
            "llama",
            &["<s>", "</s>", "<unk>"],
            None,
            None,
            None,
            Some(true),
            None,
            None,
        );
        let path = write_temp_gguf(&gguf);
        let model = Model::load(&path).unwrap();
        let tok = Tokenizer::from_gguf(&model).expect("missing id keys must not error");
        assert_eq!(tok.bos_id(), 1);
        assert_eq!(tok.eos_id(), 2);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_from_gguf_bos_eos_default_gpt2() {
        // Neither tokenizer.ggml.bos_token_id nor eos_token_id is present --
        // GPT2 defaults are bos = 0, eos = 2.
        let gguf = build_tokenizer_gguf(
            "gpt2",
            &["!", "\"", "#"],
            None,
            None,
            None,
            Some(false),
            None,
            None,
        );
        let path = write_temp_gguf(&gguf);
        let model = Model::load(&path).unwrap();
        let tok = Tokenizer::from_gguf(&model).expect("missing id keys must not error");
        assert_eq!(tok.bos_id(), 0);
        assert_eq!(tok.eos_id(), 2);
        std::fs::remove_file(path).unwrap();
    }

    // -----------------------------------------------------------------------
    // Tests: encode / decode (LLaMA)
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_decode_llama_basic() {
        // Vocab: ["<s>", "</s>", "<U+2581>hell", "o", "<U+2581>world"]
        // Real llama SPM vocabularies mark word-initial pieces with the
        // U+2581 ("<U+2581>") marker rather than a plain ASCII space, so the
        // word-initial pieces here carry it too.  Scores are all zero so
        // Viterbi picks the longest match.
        let tok = Tokenizer::new(
            TokenizerModel::LLaMA,
            vec!["<s>", "</s>", "\u{2581}hell", "o", "\u{2581}world"],
            Some(vec![0.0, 0.0, 0.0, 0.0, 0.0]),
            0,
            1,
            false,
        );
        let encoded = tok.encode("hello world").unwrap();
        let decoded = tok.decode(&encoded).unwrap();
        // decode() joins raw piece bytes without inverting SPM
        // normalization, so the marker (in place of every space, including
        // the prepended leading one) is still present in the output.
        assert_eq!(decoded, "\u{2581}hello\u{2581}world");
    }

    #[test]
    fn test_encode_decode_llama_with_bos() {
        let tok = Tokenizer::new(
            TokenizerModel::LLaMA,
            vec!["<s>", "</s>", "\u{2581}hello"],
            Some(vec![0.0, 0.0, 0.0]),
            0,
            1,
            true,
        );
        let encoded = tok.encode("hello").unwrap();
        assert_eq!(encoded.first(), Some(&0)); // BOS is first
        let decoded = tok.decode(&encoded).unwrap();
        // The decoded text starts with BOS byte sequence
        assert!(decoded.contains("hello"));
    }

    #[test]
    fn test_encode_without_bos() {
        let tok = Tokenizer::new(
            TokenizerModel::LLaMA,
            vec!["<s>", "</s>", "\u{2581}hello"],
            Some(vec![0.0, 0.0, 0.0]),
            0,
            1,
            false,
        );
        let encoded = tok.encode("hello").unwrap();
        assert_eq!(encoded, &[2]); // "<U+2581>hello" is token 2
    }

    // -----------------------------------------------------------------------
    // Tests: SPM space normalization (encode) -- the fix under test
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_llama_applies_spm_space_normalization() {
        // A vocab shaped like a real llama SPM vocab: word-initial pieces
        // carry the U+2581 marker, plus enough single characters and a
        // `<0x20>` byte-fallback entry that a *broken* encoder (one that
        // runs Viterbi over the raw, un-normalized bytes) can still produce
        // *some* -- but wrong -- tokenization instead of erroring out.
        let tok = Tokenizer::new(
            TokenizerModel::LLaMA,
            vec![
                "<s>",           // 0
                "</s>",          // 1
                "\u{2581}hello", // 2 -- word-initial "hello"
                "\u{2581}world", // 3 -- word-initial "world"
                "h",             // 4
                "e",             // 5
                "l",             // 6
                "o",             // 7
                "w",             // 8
                "r",             // 9
                "d",             // 10
                "<0x20>",        // 11 -- byte fallback for raw space
            ],
            Some(vec![
                0.0, 0.0, -10.0, -10.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
            ]),
            0,
            1,
            false,
        );

        let encoded = tok.encode("hello world").unwrap();

        // Correct SPM behavior: "hello world" normalizes to
        // "<U+2581>hello<U+2581>world" before Viterbi, which matches the two
        // whole-word pieces directly -- not the byte-fallback/char-by-char
        // path a raw-bytes Viterbi would be forced into.
        assert_eq!(
            encoded,
            &[2, 3],
            "encode must apply SPM space normalization before Viterbi \
             so word-initial marker-prefixed pieces can match"
        );
    }

    #[test]
    fn test_encode_llama_add_space_prefix_false_from_gguf_skips_leading_space() {
        // tokenizer.ggml.add_space_prefix = false must suppress the
        // leading-space prefix llama.cpp normally adds, while internal
        // spaces are still replaced with the SPM marker.
        let gguf = build_tokenizer_gguf(
            "llama",
            &["<s>", "</s>", "\u{2581}hello", "hello"],
            Some(&[0.0, 0.0, 0.0, 0.0]),
            Some(0),
            Some(1),
            Some(false),
            None,
            Some(false),
        );
        let path = write_temp_gguf(&gguf);
        let model = Model::load(&path).unwrap();
        let tok = Tokenizer::from_gguf(&model).unwrap();
        std::fs::remove_file(path).unwrap();

        // No leading space is prepended, so "hello" matches the bare piece
        // (token 3), not the marker-prefixed one (token 2).
        let encoded = tok.encode("hello").unwrap();
        assert_eq!(encoded, &[3]);
    }

    #[test]
    fn test_encode_llama_add_space_prefix_true_from_gguf_prepends_leading_space() {
        // Same vocab as above but with add_space_prefix left unset (so it
        // defaults to true for LLaMA) -- "hello" must now match the
        // marker-prefixed piece because of the prepended leading space.
        let gguf = build_tokenizer_gguf(
            "llama",
            &["<s>", "</s>", "\u{2581}hello", "hello"],
            Some(&[0.0, 0.0, 0.0, 0.0]),
            Some(0),
            Some(1),
            Some(false),
            None,
            None,
        );
        let path = write_temp_gguf(&gguf);
        let model = Model::load(&path).unwrap();
        let tok = Tokenizer::from_gguf(&model).unwrap();
        std::fs::remove_file(path).unwrap();

        let encoded = tok.encode("hello").unwrap();
        assert_eq!(encoded, &[2]);
    }

    #[test]
    fn test_encode_gpt2_unaffected_by_spm_normalization() {
        // GPT-2 / BPE must never apply SPM space normalization: `encode_gpt2`
        // pre-tokenizes on whitespace (dropping the delimiter, unchanged by
        // this fix) rather than emitting an SPM marker token -- so "a b"
        // must yield exactly the byte tokens for 'a' and 'b', with no marker
        // bytes (0xE2 0x96 0x81) anywhere in the output.
        let mut vocab: Vec<&str> = (0..256).map(|_| "").collect();
        for b in 0u8..=255 {
            let s: String = char::from(b).to_string();
            vocab[b as usize] = Box::leak(s.into_boxed_str());
        }
        let tok = Tokenizer::new(
            TokenizerModel::GPT2,
            vocab,
            Some(vec![0.0; 256]),
            0,
            1,
            false,
        );
        let encoded = tok.encode("a b").unwrap();
        assert_eq!(
            encoded,
            &[97, 98],
            "GPT-2 path must be unaffected by SPM space normalization"
        );
    }

    // -----------------------------------------------------------------------
    // Tests: empty input
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_empty_no_bos() {
        let tok = Tokenizer::new(
            TokenizerModel::LLaMA,
            vec!["<s>", "</s>"],
            Some(vec![0.0, 0.0]),
            0,
            1,
            false,
        );
        let encoded = tok.encode("").unwrap();
        assert!(encoded.is_empty());
    }

    #[test]
    fn test_encode_empty_with_bos() {
        let tok = Tokenizer::new(
            TokenizerModel::LLaMA,
            vec!["<s>", "</s>"],
            Some(vec![0.0, 0.0]),
            0,
            1,
            true,
        );
        let encoded = tok.encode("").unwrap();
        assert_eq!(encoded, &[0]); // just BOS
    }

    #[test]
    fn test_decode_empty() {
        let tok = Tokenizer::new(
            TokenizerModel::LLaMA,
            vec!["<s>", "</s>"],
            Some(vec![0.0, 0.0]),
            0,
            1,
            false,
        );
        let decoded = tok.decode(&[]).unwrap();
        assert_eq!(decoded, "");
    }

    // -----------------------------------------------------------------------
    // Tests: invalid token IDs
    // -----------------------------------------------------------------------

    #[test]
    fn test_decode_out_of_range() {
        let tok = Tokenizer::new(
            TokenizerModel::LLaMA,
            vec!["a", "b"],
            Some(vec![0.0, 0.0]),
            0,
            1,
            false,
        );
        let err = tok.decode(&[99]).unwrap_err();
        assert!(matches!(err, TokenizerError::TokenOutOfRange(99, 2)));
    }

    #[test]
    fn test_decode_detokenized_skips_special() {
        let tok = Tokenizer::new(
            TokenizerModel::LLaMA,
            vec!["<s>", "</s>", "hello"],
            Some(vec![0.0, 0.0, 0.0]),
            0,
            1,
            true,
        );
        let out = tok.decode_detokenized(&[0, 2, 1]).unwrap();
        assert_eq!(out, "hello");
    }

    // -----------------------------------------------------------------------
    // Tests: multi-byte UTF-8
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_decode_utf8() {
        // "caf\u{00E9}" is a single word, so the word-initial piece ("caf")
        // carries the SPM marker added by the mandatory leading-space
        // normalization for LLaMA.
        let tok = Tokenizer::new(
            TokenizerModel::LLaMA,
            vec!["<s>", "</s>", "\u{2581}caf", "\u{00E9}"],
            Some(vec![0.0, 0.0, 0.0, 0.0]),
            0,
            1,
            false,
        );
        let encoded = tok.encode("caf\u{00E9}").unwrap();
        let decoded = tok.decode(&encoded).unwrap();
        assert_eq!(decoded, "\u{2581}caf\u{00E9}");
    }

    // -----------------------------------------------------------------------
    // Tests: Viterbi correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_viterbi_chooses_optimal_segmentation() {
        // Vocab with known scores (log probs, negative = high prob).
        // "ab"/"a" carry the SPM word-initial marker because they are the
        // only word in the input -- "b" and "cd" are never word-initial, so
        // they stay bare, matching how real llama SPM vocabularies mark
        // only the first piece of a word.
        // Token 0: "<U+2581>ab"  score = -10.0 (very likely)
        // Token 1: "<U+2581>a"   score = -1.0  (less likely)
        // Token 2: "b"           score = -1.0  (less likely)
        let tok = Tokenizer::new(
            TokenizerModel::LLaMA,
            vec!["\u{2581}ab", "\u{2581}a", "b", "cd"],
            Some(vec![-10.0, -1.0, -1.0, -10.0]),
            0,
            1,
            false,
        );

        // For "ab":
        // Option 1: token 0 ("<U+2581>ab") cost = -(-10.0) = 10.0
        // Option 2: token 1 + token 2 ("<U+2581>a"+"b") cost = -(-1.0) + -(-1.0) = 2.0
        // Viterbi should pick option 2 because 2.0 < 10.0
        let encoded = tok.encode("ab").unwrap();
        assert_eq!(
            encoded,
            &[1, 2],
            "Viterbi should pick '<U+2581>a'+'b' over '<U+2581>ab'"
        );

        // For "abcd":
        // Option 1: "<U+2581>ab"+"cd" cost = 10.0 + 10.0 = 20.0
        // Option 2: "<U+2581>a"+"b"+"cd" cost = 1.0 + 1.0 + 10.0 = 12.0
        // Viterbi should pick option 2
        let encoded = tok.encode("abcd").unwrap();
        assert_eq!(encoded, &[1, 2, 3]);
    }

    #[test]
    fn test_viterbi_equal_scores() {
        // When scores are equal, any valid segmentation works — the longest
        // match at each position wins because DP completes at the first match.
        // Word-initial pieces carry the SPM marker (added by encode's
        // mandatory leading-space normalization for LLaMA).
        let tok = Tokenizer::new(
            TokenizerModel::LLaMA,
            vec!["<s>", "</s>", "\u{2581}hello", "\u{2581}hell", "o"],
            Some(vec![0.0, 0.0, 0.0, 0.0, 0.0]),
            0,
            1,
            false,
        );
        // "hello" can be token 2 or token 3+4
        // With equal scores, Viterbi picks the first reachable option
        let encoded = tok.encode("hello").unwrap();
        // Both are valid -- just check it decodes back correctly. decode()
        // joins raw piece bytes without inverting SPM normalization, so the
        // leading marker survives into the decoded text.
        let decoded = tok.decode(&encoded).unwrap();
        assert_eq!(decoded, "\u{2581}hello");
    }

    // -----------------------------------------------------------------------
    // Tests: byte fallback
    // -----------------------------------------------------------------------

    #[test]
    fn test_byte_fallback_unknown_bytes() {
        // Tokenizer with byte-fallback tokens for 'a', 'b', and the three
        // bytes of the SPM marker (U+2581 = 0xE2 0x96 0x81) -- the mandatory
        // leading-space prefix normalizes to that marker before Viterbi, and
        // this vocab has no dedicated marker piece, so it must fall back to
        // per-byte tokens for it too.
        let tok = Tokenizer::new(
            TokenizerModel::LLaMA,
            vec![
                "<s>", "</s>", "<0xE2>", // marker byte 1
                "<0x96>", // marker byte 2
                "<0x81>", // marker byte 3
                "<0x61>", // 'a'
                "<0x62>", // 'b'
            ],
            Some(vec![0.0, 0.0, -5.0, -5.0, -5.0, -5.0, -5.0]),
            0,
            1,
            false,
        );
        let encoded = tok.encode("ab").unwrap();
        // Each byte (marker bytes, then 'a', then 'b') maps to its
        // byte-fallback token.
        assert_eq!(encoded, &[2, 3, 4, 5, 6]);
        let decoded = tok.decode(&encoded).unwrap();
        // Decode returns the literal token strings, not the raw bytes.
        assert_eq!(decoded, "<0xE2><0x96><0x81><0x61><0x62>");
    }

    // -----------------------------------------------------------------------
    // Tests: GPT-2 BPE
    // -----------------------------------------------------------------------

    #[test]
    fn test_bpe_encode_simple_word() {
        // For a GPT-2 tokenizer, byte tokens are at IDs matching their byte
        // value.  The merge table says: token(104) + token(105) = "hi" is a
        // merge -> merged into a single token.  Let's set up a minimal vocab.
        //
        // tokens[0..256] are byte tokens (identity mapping).
        // tokens[256] = "he" (merge of token(104) and token(101))
        // tokens[257] = "hello" (merge of "he" and "llo")
        // But this is already getting complex for a unit test.
        //
        // Simpler: build a tiny vocab with explicit byte mappings.
        let mut vocab: Vec<&str> = (0..256).map(|_| "").collect();
        // Fill in byte tokens
        for b in 0u8..=255 {
            let ch = char::from(b);
            let s: String = ch.to_string();
            vocab[b as usize] = Box::leak(s.into_boxed_str());
        }

        // Add merged tokens
        // "a" = token 97, "b" = token 98
        // Merge "ab" -> new token 256
        let tok = Tokenizer::new(
            TokenizerModel::GPT2,
            vocab,
            Some(vec![0.0; 257]),
            0,
            1,
            false,
        );

        // We can't really test BPE without merges — just test that encoding
        // produces byte-level tokens for simple input.
        let encoded = tok.encode("a").unwrap();
        assert_eq!(encoded, &[97]); // 'a' is byte value 97
    }

    #[test]
    fn test_bpe_with_merges() {
        // Build a 257-token vocabulary: byte tokens 0–255, then token 256 = "ab"
        // For bytes that are not valid UTF-8 individually, use lossy conversion.
        let mut token_strs: Vec<String> = Vec::with_capacity(257);
        for b in 0u8..=255 {
            let s = String::from_utf8_lossy(&[b]).to_string();
            token_strs.push(s);
        }
        token_strs.push("ab".to_string());

        // Convert to &str slice for the GGUF builder
        let token_refs: Vec<&str> = token_strs.iter().map(String::as_str).collect();

        // Build GGUF with one merge rule: "a b"
        let gguf = build_tokenizer_gguf(
            "gpt2",
            &token_refs,
            None,
            Some(0),
            Some(1),
            Some(false),
            Some(&["a b"]),
            None,
        );
        let path = write_temp_gguf(&gguf);
        let model = Model::load(&path).unwrap();
        let tok = Tokenizer::from_gguf(&model).unwrap();
        std::fs::remove_file(path).unwrap();

        // Encode "ab" — should merge byte tokens 97 and 98 into token 256
        let encoded = tok.encode("ab").unwrap();
        assert_eq!(encoded, &[256], "BPE should merge 'a'+'b' into one token");
    }

    // -----------------------------------------------------------------------
    // Tests: tokenizer using new() helper round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_tokenizer_roundtrip() {
        // Word-initial pieces carry the SPM marker so the mandatory
        // leading-space normalization for LLaMA can match them.
        let tok = Tokenizer::new(
            TokenizerModel::LLaMA,
            vec!["<s>", "</s>", "\u{2581}hello", "\u{2581}world"],
            Some(vec![0.0; 4]),
            0,
            1,
            false,
        );
        let text = "hello world";
        let encoded = tok.encode(text).unwrap();
        let decoded = tok.decode(&encoded).unwrap();
        // decode() joins raw piece bytes without inverting SPM
        // normalization, so the decoded text carries the marker in place of
        // every space (including the prepended leading one) rather than
        // matching the original text byte-for-byte.
        assert_eq!(decoded, "\u{2581}hello\u{2581}world");
    }

    // -----------------------------------------------------------------------
    // Tests: non-ASCII
    // -----------------------------------------------------------------------

    #[test]
    fn test_non_ascii_roundtrip() {
        // "\u{00E9}\u{2603}" is a single word (no internal space), so only
        // the first piece ("\u{00E9}") carries the SPM marker.
        let tok = Tokenizer::new(
            TokenizerModel::LLaMA,
            vec!["<s>", "</s>", "\u{2581}\u{00E9}", "\u{2603}"], // <U+2581> + e-acute and snowman
            Some(vec![0.0; 4]),
            0,
            1,
            false,
        );
        let text = "\u{00E9}\u{2603}";
        let encoded = tok.encode(text).unwrap();
        let decoded = tok.decode(&encoded).unwrap();
        assert_eq!(decoded, "\u{2581}\u{00E9}\u{2603}");
    }

    // -----------------------------------------------------------------------
    // Tests: decode_detokenized
    // -----------------------------------------------------------------------

    #[test]
    fn test_decode_detokenized_filters_bos_eos() {
        let tok = Tokenizer::new(
            TokenizerModel::LLaMA,
            vec!["<s>", "</s>", "hello"],
            None,
            0,
            1,
            true,
        );
        let out = tok.decode_detokenized(&[0, 2, 1]).unwrap();
        assert_eq!(out, "hello");
    }

    // -----------------------------------------------------------------------
    // Tests: hex_nibble
    // -----------------------------------------------------------------------

    #[test]
    fn test_hex_nibble_valid() {
        assert_eq!(hex_nibble(b'0'), Some(0));
        assert_eq!(hex_nibble(b'9'), Some(9));
        assert_eq!(hex_nibble(b'a'), Some(10));
        assert_eq!(hex_nibble(b'f'), Some(15));
        assert_eq!(hex_nibble(b'A'), Some(10));
        assert_eq!(hex_nibble(b'F'), Some(15));
    }

    #[test]
    fn test_hex_nibble_invalid() {
        assert_eq!(hex_nibble(b'g'), None);
        assert_eq!(hex_nibble(b'z'), None);
        assert_eq!(hex_nibble(b'@'), None);
        assert_eq!(hex_nibble(b' '), None);
    }

    // -----------------------------------------------------------------------
    // Tests: GGUFValueExt::as_u32_lossy
    // -----------------------------------------------------------------------

    #[test]
    fn as_u32_lossy_uint64_in_range() {
        assert_eq!(GGUFValue::Uint64(7).as_u32_lossy().unwrap(), 7);
    }

    #[test]
    fn as_u32_lossy_int64_in_range() {
        assert_eq!(GGUFValue::Int64(7).as_u32_lossy().unwrap(), 7);
    }

    #[test]
    fn as_u32_lossy_int32_negative_is_invalid_id() {
        let err = GGUFValue::Int32(-1).as_u32_lossy().unwrap_err();
        assert!(matches!(err, TokenizerError::InvalidId(-1)));
    }

    #[test]
    fn as_u32_lossy_uint64_out_of_range_is_err() {
        let too_big = u64::from(u32::MAX) + 1;
        let err = GGUFValue::Uint64(too_big).as_u32_lossy().unwrap_err();
        assert!(matches!(err, TokenizerError::MissingMetadata(_)));
    }

    #[test]
    fn as_u32_lossy_non_integer_variant_is_err() {
        let err = GGUFValue::String("x".to_string())
            .as_u32_lossy()
            .unwrap_err();
        assert!(matches!(err, TokenizerError::MissingMetadata(_)));
    }
}
