//! The search engine: owns the corpus + both indexes, answers queries.
//!
//! Contract:
//! - `Engine::build(cfg)` loads chunks, builds BM25, and (unless
//!   `cfg.lexical_only`) loads the embedder and embeds every chunk's
//!   `indexable_text()` into the vector index, going through
//!   [`crate::cache::EmbeddingCache`] so only new/changed chunks hit the model.
//!   Logs timing per phase.
//! - `search(query, k)` runs BM25 (top `4*k`) and dense (top `4*k`), fuses
//!   with [`crate::hybrid::fuse`], applies the phrase bonus, and returns
//!   `SearchHit`s with a `snippet` (first ~400 chars of the chunk).
//! - `get_chunk(id)` / `get_document(path)` return full text for follow-up.

use std::path::Path;

use crate::{bm25::Bm25Index, config::Config, corpus::Chunk, embed::Embedder, vector::VectorIndex};

#[derive(Debug, Clone, serde::Serialize)]
pub struct SearchHit {
    pub chunk_id: usize,
    pub path: String,
    pub heading_path: Vec<String>,
    pub line_start: usize,
    pub line_end: usize,
    pub score: f32,
    pub snippet: String,
}

pub struct Engine {
    pub cfg: Config,
    pub chunks: Vec<Chunk>,
    pub bm25: Bm25Index,
    pub embedder: Option<Embedder>,
    pub vectors: VectorIndex,
}

impl Engine {
    pub fn build(cfg: Config) -> anyhow::Result<Self> {
        let _ = cfg;
        todo!()
    }

    pub fn search(&mut self, query: &str, k: usize) -> anyhow::Result<Vec<SearchHit>> {
        let _ = (query, k);
        todo!()
    }

    pub fn get_chunk(&self, id: usize) -> Option<&Chunk> {
        self.chunks.get(id)
    }

    /// Full text of every chunk from `path`, in order, joined with blank lines.
    pub fn get_document(&self, path: &Path) -> Option<String> {
        let _ = path;
        todo!()
    }

    pub fn stats(&self) -> serde_json::Value {
        let _ = self;
        todo!("{{ files, chunks, lexical_terms, dense: bool, dim }}")
    }
}
