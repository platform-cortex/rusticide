// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;

use rusticide_core::Tensor;
use rusticide_io::{GgufFile, Metadata};

use crate::error::{ModelError, Result};

/// Where a model reads its metadata and tensors from.
pub trait ModelSource {
    fn metadata(&self) -> &Metadata;
    fn tensor_shape(&self, name: &str) -> Option<Vec<usize>>;
    /// # Errors
    ///
    /// `ModelError::MissingTensor` when `name` is absent; loader errors otherwise.
    fn tensor(&self, name: &str) -> Result<Tensor>;
}

impl ModelSource for GgufFile {
    fn metadata(&self) -> &Metadata {
        Self::metadata(self)
    }

    fn tensor_shape(&self, name: &str) -> Option<Vec<usize>> {
        self.tensor_info(name).map(|info| info.shape.clone())
    }

    fn tensor(&self, name: &str) -> Result<Tensor> {
        if self.tensor_info(name).is_none() {
            return Err(ModelError::MissingTensor {
                name: name.to_owned(),
            });
        }
        Ok(Self::tensor(self, name)?)
    }
}

/// A source built from tensors held in memory, for tests and fixtures.
#[derive(Debug)]
pub struct InMemorySource {
    metadata: Metadata,
    tensors: HashMap<String, Tensor>,
}

impl InMemorySource {
    #[must_use]
    pub fn new(metadata: Metadata) -> Self {
        Self {
            metadata,
            tensors: HashMap::new(),
        }
    }

    #[must_use]
    pub fn with_tensor(mut self, name: &str, tensor: Tensor) -> Self {
        self.tensors.insert(name.to_owned(), tensor);
        self
    }
}

impl ModelSource for InMemorySource {
    fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    fn tensor_shape(&self, name: &str) -> Option<Vec<usize>> {
        self.tensors.get(name).map(|t| t.shape().dims().to_vec())
    }

    fn tensor(&self, name: &str) -> Result<Tensor> {
        self.tensors
            .get(name)
            .cloned()
            .ok_or_else(|| ModelError::MissingTensor {
                name: name.to_owned(),
            })
    }
}

#[cfg(test)]
mod tests {
    use rusticide_core::Tensor;
    use rusticide_io::{GgufFile, Metadata};

    use super::{InMemorySource, ModelSource};
    use crate::error::ModelError;

    fn source() -> InMemorySource {
        let tensor = Tensor::from_slice(&[1.0_f32, 2.0], &[2]).unwrap();
        InMemorySource::new(Metadata::default()).with_tensor("a", tensor)
    }

    #[test]
    fn in_memory_source_returns_its_tensors() {
        let source = source();
        assert!(source.metadata().is_empty());
        assert_eq!(source.tensor_shape("a"), Some(vec![2]));
        assert_eq!(source.tensor_shape("b"), None);
        assert_eq!(
            source.tensor("a").unwrap().to_vec::<f32>().unwrap(),
            [1.0, 2.0]
        );
    }

    #[test]
    fn in_memory_source_reports_a_missing_tensor() {
        match source().tensor("nope") {
            Err(ModelError::MissingTensor { name }) => assert_eq!(name, "nope"),
            other => panic!("unexpected: {other:?}"),
        }
    }

    fn gguf_with_one_tensor() -> GgufFile {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.push(b'w');
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.resize(72, 0);
        bytes[64..68].copy_from_slice(&1.0_f32.to_le_bytes());
        bytes[68..72].copy_from_slice(&2.0_f32.to_le_bytes());
        GgufFile::from_bytes(bytes).unwrap()
    }

    #[test]
    fn gguf_source_reads_present_and_reports_missing_tensors() {
        let file = gguf_with_one_tensor();
        let source: &dyn ModelSource = &file;
        assert!(source.metadata().is_empty());
        assert_eq!(source.tensor_shape("w"), Some(vec![2]));
        assert_eq!(source.tensor_shape("nope"), None);
        assert_eq!(
            source.tensor("w").unwrap().to_vec::<f32>().unwrap(),
            [1.0, 2.0]
        );
        match source.tensor("nope") {
            Err(ModelError::MissingTensor { name }) => assert_eq!(name, "nope"),
            other => panic!("unexpected: {other:?}"),
        }
    }
}
