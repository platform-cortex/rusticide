// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;

use rusticide_io::{GGUFValue, Metadata};

use crate::error::{Result, TokenizerError};
use crate::model::TokenizerModel;
use crate::tokenizer::{MergeLookup, Tokenizer, pad_scores, token_id};
use crate::value_ext::GGUFValueExt;

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
        .map(GGUFValueExt::as_string)
        .collect()
}

fn read_scores(metadata: &Metadata) -> Result<Vec<f32>> {
    let Some(value) = metadata.get("tokenizer.ggml.scores") else {
        return Ok(Vec::new());
    };
    Ok(value
        .as_array()?
        .iter()
        .map(|v| v.as_f32().unwrap_or(0.0))
        .collect())
}

fn read_id(metadata: &Metadata, key: &str, default: u32) -> Result<u32> {
    metadata
        .get(key)
        .map_or(Ok(default), GGUFValueExt::as_u32_lossy)
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

impl Tokenizer {
    /// Construct a tokenizer from GGUF model metadata.
    ///
    /// Reads `tokenizer.ggml.*` keys from the metadata store.
    ///
    /// # Errors
    ///
    /// Returns `MissingMetadata` if required keys are missing or have the
    /// wrong type, `UnknownModel` if the model type is not recognized.
    pub fn from_gguf(metadata: &Metadata) -> Result<Self> {
        let model = read_model(metadata)?;
        let vocab_strings = read_tokens(metadata)?;
        let scores = read_scores(metadata)?;
        let is_llama = model == TokenizerModel::LLaMA;
        let bos_id = read_id(metadata, "tokenizer.ggml.bos_token_id", u32::from(is_llama))?;
        let eos_id = read_id(metadata, "tokenizer.ggml.eos_token_id", 2)?;
        let add_bos = read_flag(metadata, "tokenizer.ggml.add_bos_token", is_llama);
        let add_space_prefix = read_flag(metadata, "tokenizer.ggml.add_space_prefix", is_llama);
        let merge_info = read_merges(metadata, model, &vocab_strings)?;
        Ok(Self {
            model,
            scores: pad_scores(scores, vocab_strings.len()),
            vocab: vocab_strings.into_iter().map(String::into_bytes).collect(),
            bos_id,
            eos_id,
            add_bos,
            add_space_prefix,
            merge_info,
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
