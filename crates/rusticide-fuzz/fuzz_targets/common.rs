// SPDX-FileCopyrightText: 2026 XLLM contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared helper for the GGUF fuzz targets.
//!
//! `Model::load` only accepts a path, so every target has to spill its input
//! to disk first. This keeps that (and the cleanup) in one place.

use std::fs;
use std::path::PathBuf;

/// A temp file that deletes itself on drop, including on panic unwind.
pub struct ScratchFile {
    path: PathBuf,
}

impl ScratchFile {
    /// Write `data` to a uniquely named temp file.
    ///
    /// The name carries the pid so parallel libFuzzer workers, which are
    /// separate processes, never collide on the same path.
    pub fn new(tag: &str, data: &[u8]) -> std::io::Result<Self> {
        let path =
            std::env::temp_dir().join(format!("rusticide-fuzz-{tag}-{}.gguf", std::process::id()));
        fs::write(&path, data)?;
        Ok(Self { path })
    }

    pub fn path(&self) -> &PathBuf {
        &self.path
    }
}

impl Drop for ScratchFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}
