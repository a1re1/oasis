//! Corpus ingestion: walk roots, read markdown, split into heading-aware chunks.
//!
//! Contract:
//! - `load(roots, cfg) -> Vec<Chunk>` walks every root recursively, skipping
//!   hidden dirs, `node_modules`, and `target`.
//! - Chunking is heading-aware: each ATX heading (`#`..`######`) starts a new
//!   section; `heading_path` is the stack of ancestor headings (e.g.
//!   `["Tokio", "Runtime", "Spawning"]`). Sections longer than
//!   `cfg.chunk_tokens` (whitespace tokens, approximate) are split further with
//!   `cfg.chunk_overlap` overlap. Fenced code blocks are never split mid-fence.
//! - `Chunk::id` is dense and stable for the lifetime of an index (`0..n`).

use std::path::{Path, PathBuf};

use crate::config::Config;

#[derive(Debug, Clone, serde::Serialize)]
pub struct Chunk {
    pub id: usize,
    /// Path relative to the root it was found under.
    pub path: PathBuf,
    /// Root directory the file was found under.
    pub root: PathBuf,
    /// Ancestor heading titles, outermost first.
    pub heading_path: Vec<String>,
    /// Chunk body (markdown source, trimmed).
    pub text: String,
    /// 1-based line range in the source file.
    pub line_start: usize,
    pub line_end: usize,
}

impl Chunk {
    /// Text handed to the embedder / tokenizer: headings + body, so that a
    /// chunk that only says "see above" still carries its context.
    pub fn indexable_text(&self) -> String {
        todo!("prefix heading_path joined by ' > ' followed by newline and text")
    }
}

/// Walk `roots` and return all chunks. Files that fail to read are logged and skipped.
pub fn load(roots: &[PathBuf], cfg: &Config) -> anyhow::Result<Vec<Chunk>> {
    let _ = (roots, cfg);
    todo!()
}

/// Chunk a single markdown document. Exposed for tests.
pub fn chunk_markdown(root: &Path, rel_path: &Path, source: &str, cfg: &Config) -> Vec<Chunk> {
    let _ = (root, rel_path, source, cfg);
    todo!()
}
