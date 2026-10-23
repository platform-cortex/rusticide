// SPDX-FileCopyrightText: 2026 Elio Severo Junior and contributors
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::{ModelError, Result};
use crate::model::Model;
use crate::source::ModelSource;

/// Builds a model of one architecture from a source.
pub type Loader = fn(&dyn ModelSource) -> Result<Box<dyn Model>>;

/// Maps `general.architecture` names to loaders.
#[derive(Default)]
pub struct Registry {
    entries: Vec<(&'static str, Loader)>,
}

impl Registry {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    #[must_use]
    pub fn with(mut self, architecture: &'static str, loader: Loader) -> Self {
        self.entries.push((architecture, loader));
        self
    }

    #[must_use]
    pub fn architectures(&self) -> Vec<&'static str> {
        self.entries.iter().map(|(name, _)| *name).collect()
    }

    /// # Errors
    ///
    /// `MissingMetadata` without `general.architecture`; `UnknownArchitecture`
    /// when no loader is registered for it; loader errors otherwise.
    pub fn load(&self, source: &dyn ModelSource) -> Result<Box<dyn Model>> {
        let name = source
            .metadata()
            .architecture()
            .ok_or_else(|| ModelError::MissingMetadata {
                key: "general.architecture".to_owned(),
            })?;
        let (_, loader) = self
            .entries
            .iter()
            .find(|(architecture, _)| *architecture == name)
            .ok_or_else(|| ModelError::UnknownArchitecture {
                name: name.to_owned(),
                supported: self.architectures().join(", "),
            })?;
        loader(source)
    }
}

#[cfg(test)]
#[path = "registry_tests.rs"]
mod tests;
