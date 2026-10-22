// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::value::GGUFValue;

const DEFAULT_ALIGNMENT: usize = 32;

struct TensorSpec {
    name: String,
    dims: Vec<u64>,
    raw_type: u32,
    data: Vec<u8>,
}

/// Fluent builder of GGUF byte images for tests.
pub(crate) struct GgufBuilder {
    magic: [u8; 4],
    version: u32,
    kvs: Vec<Vec<u8>>,
    tensors: Vec<TensorSpec>,
    alignment: usize,
}

pub(crate) fn string_bytes(text: &str) -> Vec<u8> {
    let mut bytes = (text.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(text.as_bytes());
    bytes
}

pub(crate) fn header_bytes(version: u32, tensor_count: u64, kv_count: u64) -> Vec<u8> {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&version.to_le_bytes());
    bytes.extend_from_slice(&tensor_count.to_le_bytes());
    bytes.extend_from_slice(&kv_count.to_le_bytes());
    bytes
}

pub(crate) fn tensor_info_bytes(name: &str, dims: &[u64], raw_type: u32, offset: u64) -> Vec<u8> {
    let mut bytes = string_bytes(name);
    bytes.extend_from_slice(&u32::try_from(dims.len()).unwrap().to_le_bytes());
    for dim in dims {
        bytes.extend_from_slice(&dim.to_le_bytes());
    }
    bytes.extend_from_slice(&raw_type.to_le_bytes());
    bytes.extend_from_slice(&offset.to_le_bytes());
    bytes
}

fn value_bytes(value: &GGUFValue) -> (u32, Vec<u8>) {
    match value {
        GGUFValue::Uint32(n) => (4, n.to_le_bytes().to_vec()),
        GGUFValue::Int32(n) => (5, n.to_le_bytes().to_vec()),
        GGUFValue::Float32(n) => (6, n.to_le_bytes().to_vec()),
        GGUFValue::String(text) => (8, string_bytes(text)),
        GGUFValue::Uint64(n) => (10, n.to_le_bytes().to_vec()),
        other => unimplemented!("test builder cannot encode {other:?}"),
    }
}

impl GgufBuilder {
    pub(crate) fn new() -> Self {
        Self {
            magic: *b"GGUF",
            version: 3,
            kvs: Vec::new(),
            tensors: Vec::new(),
            alignment: DEFAULT_ALIGNMENT,
        }
    }

    #[must_use]
    pub(crate) fn magic(mut self, magic: [u8; 4]) -> Self {
        self.magic = magic;
        self
    }

    #[must_use]
    pub(crate) fn version(mut self, version: u32) -> Self {
        self.version = version;
        self
    }

    #[must_use]
    pub(crate) fn kv(self, key: &str, value: &GGUFValue) -> Self {
        let (tag, payload) = value_bytes(value);
        self.raw_kv(key, tag, &payload)
    }

    #[must_use]
    pub(crate) fn raw_kv(mut self, key: &str, tag: u32, payload: &[u8]) -> Self {
        let mut bytes = string_bytes(key);
        bytes.extend_from_slice(&tag.to_le_bytes());
        bytes.extend_from_slice(payload);
        self.kvs.push(bytes);
        self
    }

    #[must_use]
    pub(crate) fn tensor(mut self, name: &str, dims: &[u64], raw_type: u32, data: &[u8]) -> Self {
        self.tensors.push(TensorSpec {
            name: name.to_owned(),
            dims: dims.to_vec(),
            raw_type,
            data: data.to_vec(),
        });
        self
    }

    /// Sets `general.alignment` (a valid `Uint32`) and pads data to match.
    #[must_use]
    pub(crate) fn alignment(mut self, alignment: u32) -> Self {
        self.alignment = usize::try_from(alignment).unwrap();
        self.kv("general.alignment", &GGUFValue::Uint32(alignment))
    }

    pub(crate) fn build(self) -> Vec<u8> {
        let mut out = header_bytes(
            self.version,
            self.tensors.len() as u64,
            self.kvs.len() as u64,
        );
        out[..4].copy_from_slice(&self.magic);
        self.kvs.iter().for_each(|kv| out.extend_from_slice(kv));
        let mut offset = 0_usize;
        for tensor in &self.tensors {
            out.extend_from_slice(&tensor_info_bytes(
                &tensor.name,
                &tensor.dims,
                tensor.raw_type,
                offset as u64,
            ));
            offset = (offset + tensor.data.len()).next_multiple_of(self.alignment);
        }
        out.resize(out.len().next_multiple_of(self.alignment), 0);
        for tensor in &self.tensors {
            out.extend_from_slice(&tensor.data);
            out.resize(out.len().next_multiple_of(self.alignment), 0);
        }
        out
    }
}
