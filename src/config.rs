//! Runtime configuration (CLI flags + env), shared by `index`, `search`, `serve`.

use std::path::PathBuf;

/// Embedding model to pull from the HF hub. Must ship `onnx/model.onnx` + `tokenizer.json`.
pub const DEFAULT_MODEL_REPO: &str = "BAAI/bge-small-en-v1.5";
/// bge-small hidden size.
pub const DEFAULT_EMBED_DIM: usize = 384;

#[derive(Debug, Clone)]
pub struct Config {
    /// Root directories to ingest (recursively, `*.md` / `*.mdx` / `*.markdown`).
    pub roots: Vec<PathBuf>,
    /// HF repo id for the ONNX embedding model.
    pub model_repo: String,
    /// Local override: directory containing `model.onnx` and `tokenizer.json`.
    pub model_dir: Option<PathBuf>,
    /// Target chunk size in tokens (approximate; chunker splits on headings first).
    pub chunk_tokens: usize,
    /// Overlap between consecutive chunks of the same section, in tokens.
    pub chunk_overlap: usize,
    /// Number of results returned by `search` when the caller does not specify.
    pub default_top_k: usize,
    /// Skip dense indexing entirely (BM25 only). Useful for tests / no-network.
    pub lexical_only: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            roots: Vec::new(),
            model_repo: DEFAULT_MODEL_REPO.to_string(),
            model_dir: None,
            chunk_tokens: 256,
            chunk_overlap: 32,
            default_top_k: 8,
            lexical_only: false,
        }
    }
}
