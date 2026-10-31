// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;

use rusticide_io::{GGUFValue, Metadata};

use crate::end_of_generation::end_of_generation_ids;
use crate::error::{Result, TokenizerError};
use crate::model::TokenizerModel;
use crate::pre_tokenizer::PreTokenizer;
use crate::special::special_tokens;
use crate::tokenizer::{MergeLookup, Tokenizer, byte_tokens_for, pad_scores, token_id};
use crate::trie::Trie;
use crate::value_ext::GGUFValueExt;

pub(crate) const MAX_TOKEN_BYTES: usize = 1024;

fn required<'a>(metadata: &'a Metadata, key: &'static str) -> Result<&'a GGUFValue> {
    metadata
        .get(key)
        .ok_or(TokenizerError::MissingMetadata(key))
}

fn read_model(metadata: &Metadata) -> Result<TokenizerModel> {
    match required(metadata, "tokenizer.ggml.model")?
        .as_string()?
        .as_str()
    {
        "llama" => Ok(TokenizerModel::LLaMA),
        "gpt2" => Ok(TokenizerModel::GPT2),
        other => Err(TokenizerError::UnknownModel(other.to_string())),
    }
}

fn read_tokens(metadata: &Metadata) -> Result<Vec<String>> {
    required(metadata, "tokenizer.ggml.tokens")?
        .as_array()?
        .iter()
        .enumerate()
        .map(|(index, value)| bounded_token(index, value.as_string()?))
        .collect()
}

fn bounded_token(index: usize, token: String) -> Result<String> {
    if token.len() > MAX_TOKEN_BYTES {
        return Err(TokenizerError::TokenTooLong {
            index,
            len: token.len(),
            max: MAX_TOKEN_BYTES,
        });
    }
    Ok(token)
}

fn read_scores(metadata: &Metadata) -> Result<Vec<f32>> {
    let Some(value) = metadata.get("tokenizer.ggml.scores") else {
        return Ok(Vec::new());
    };
    value
        .as_array()?
        .iter()
        .enumerate()
        .map(|(index, v)| match v.as_f32() {
            Ok(score) if !score.is_finite() => Err(TokenizerError::NonFiniteScore { index }),
            Ok(score) => Ok(score),
            Err(_) => Ok(0.0),
        })
        .collect()
}

fn read_special_tokens(metadata: &Metadata, vocab: &[String]) -> Result<Vec<(String, u32)>> {
    metadata
        .get("tokenizer.ggml.token_type")
        .map_or(Ok(Vec::new()), |types| {
            Ok(special_tokens(vocab, types.as_array()?))
        })
}

fn read_id(metadata: &Metadata, key: &str, default: u32) -> Result<u32> {
    metadata
        .get(key)
        .map_or(Ok(default), GGUFValueExt::as_u32_lossy)
}

fn read_optional_id(metadata: &Metadata, key: &str) -> Result<Option<u32>> {
    metadata
        .get(key)
        .map(GGUFValueExt::as_u32_lossy)
        .transpose()
}

fn read_flag(metadata: &Metadata, key: &str, default: bool) -> bool {
    metadata
        .get(key)
        .map_or(default, |v| v.as_bool().unwrap_or(default))
}

fn read_merges(
    metadata: &Metadata,
    model: TokenizerModel,
    vocab: &[String],
) -> Result<MergeLookup> {
    let Some(merges) = metadata.get("tokenizer.ggml.merges") else {
        return Ok(HashMap::new());
    };
    if model != TokenizerModel::GPT2 {
        return Ok(HashMap::new());
    }
    let token_to_id: HashMap<String, u32> = vocab
        .iter()
        .enumerate()
        .map(|(index, token)| Ok((token.clone(), token_id(index)?)))
        .collect::<Result<_>>()?;
    Tokenizer::build_merge_lookup(&token_to_id, merges.as_array()?)
}

fn read_pre_tokenizer(metadata: &Metadata, model: TokenizerModel) -> Result<Option<PreTokenizer>> {
    if model != TokenizerModel::GPT2 {
        return Ok(None);
    }
    let name = metadata
        .get("tokenizer.ggml.pre")
        .map(GGUFValueExt::as_string)
        .transpose()?;
    PreTokenizer::from_name(name.as_deref()).map(Some)
}

fn require_byte_tokens(model: TokenizerModel, byte_tokens: &[Option<u32>; 256]) -> Result<()> {
    if model != TokenizerModel::GPT2 {
        return Ok(());
    }
    (0..=255u8)
        .find(|&byte| byte_tokens[usize::from(byte)].is_none())
        .map_or(Ok(()), |byte| {
            Err(TokenizerError::MissingByteToken { byte })
        })
}

impl Tokenizer {
    /// Construct a tokenizer from GGUF model metadata.
    ///
    /// Reads `tokenizer.ggml.*` keys from the metadata store.
    ///
    /// # Errors
    ///
    /// Returns `MissingMetadata` if required keys are missing or have the
    /// wrong type, `UnknownModel` if the model type is not recognized,
    /// `MissingByteToken` if a GPT-2 vocabulary lacks a byte's token,
    /// `UnsupportedPreTokenizer` if `tokenizer.ggml.pre` names an unknown
    /// pre-tokenizer.
    pub fn from_gguf(metadata: &Metadata) -> Result<Self> {
        let model = read_model(metadata)?;
        let vocab_strings = read_tokens(metadata)?;
        let scores = read_scores(metadata)?;
        let is_llama = model == TokenizerModel::LLaMA;
        let bos_id = read_id(metadata, "tokenizer.ggml.bos_token_id", u32::from(is_llama))?;
        let eos_id = read_id(metadata, "tokenizer.ggml.eos_token_id", 2)?;
        let end_of_turn_id = read_optional_id(metadata, "tokenizer.ggml.eot_token_id")?;
        let add_bos = read_flag(metadata, "tokenizer.ggml.add_bos_token", is_llama);
        let add_space_prefix = read_flag(metadata, "tokenizer.ggml.add_space_prefix", is_llama);
        let merge_info = read_merges(metadata, model, &vocab_strings)?;
        let special_tokens = read_special_tokens(metadata, &vocab_strings)?;
        let vocab: Vec<Vec<u8>> = vocab_strings.into_iter().map(String::into_bytes).collect();
        let byte_tokens = byte_tokens_for(model, &vocab);
        require_byte_tokens(model, &byte_tokens)?;
        let pre = read_pre_tokenizer(metadata, model)?;
        Ok(Self {
            end_of_generation: end_of_generation_ids(&vocab, eos_id, end_of_turn_id),
            model,
            byte_tokens,
            pre,
            scores: pad_scores(scores, vocab.len()),
            pieces: Trie::from_vocab(&vocab),
            vocab,
            bos_id,
            eos_id,
            add_bos,
            add_space_prefix,
            merge_info,
            special_tokens,
        })
    }

    pub(crate) fn build_merge_lookup(
        token_to_id: &HashMap<String, u32>,
        merges: &[GGUFValue],
    ) -> Result<MergeLookup> {
        let mut merge_info = HashMap::new();
        for (rank, merge_val) in merges.iter().enumerate() {
            let pair = merge_val.as_string()?;
            let Some((left, right)) = pair.split_once(' ') else {
                continue;
            };
            let ids = (
                token_to_id.get(&format!("{left}{right}")),
                token_to_id.get(left),
                token_to_id.get(right),
            );
            if let (Some(&merged), Some(&left_id), Some(&right_id)) = ids {
                merge_info.insert((left_id, right_id), (rank, merged));
            }
        }
        Ok(merge_info)
    }
}

#[cfg(test)]
#[path = "from_gguf_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "merge_tests.rs"]
mod merge_tests;

#[cfg(test)]
#[path = "token_limit_tests.rs"]
mod token_limit_tests;

#[cfg(test)]
#[path = "byte_token_tests.rs"]
mod byte_token_tests;

#[cfg(test)]
#[path = "pre_gguf_tests.rs"]
mod pre_gguf_tests;
