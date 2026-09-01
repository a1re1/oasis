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
use std::time::Instant;

use crate::{
    bm25::Bm25Index,
    cache::EmbeddingCache,
    config::Config,
    corpus::{self, Chunk},
    embed::Embedder,
    hybrid::{self, PHRASE_BONUS, Weights},
    vector::VectorIndex,
};

pub const SNIPPET_CHARS: usize = 400;

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
        let t = Instant::now();
        let chunks = corpus::load(&cfg.roots, &cfg)?;
        tracing::info!(
            chunks = chunks.len(),
            ms = t.elapsed().as_millis(),
            "loaded corpus"
        );

        let t = Instant::now();
        let bm25 = Bm25Index::build(&chunks);
        tracing::info!(
            terms = bm25.term_count(),
            ms = t.elapsed().as_millis(),
            "built bm25"
        );

        let mut engine = Self {
            cfg,
            chunks,
            bm25,
            embedder: None,
            vectors: VectorIndex::new(0),
        };
        if engine.cfg.lexical_only {
            return Ok(engine);
        }

        let t = Instant::now();
        let mut embedder = Embedder::load(&engine.cfg)?;
        tracing::info!(ms = t.elapsed().as_millis(), "loaded embedder");

        let t = Instant::now();
        let mut cache = EmbeddingCache::load(&engine.cfg);
        let texts: Vec<String> = engine.chunks.iter().map(Chunk::indexable_text).collect();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let vecs = cache.get_or_embed(&mut embedder, &refs)?;
        let dim = vecs.first().map_or(embedder.dim(), Vec::len);
        let mut vectors = VectorIndex::new(dim);
        for v in &vecs {
            vectors.push(v);
        }
        if let Err(e) = cache.save() {
            tracing::warn!(error = %e, "failed to save embedding cache");
        }
        tracing::info!(
            vectors = vectors.len(),
            dim,
            ms = t.elapsed().as_millis(),
            "built dense index"
        );

        engine.embedder = Some(embedder);
        engine.vectors = vectors;
        Ok(engine)
    }

    pub fn search(&mut self, query: &str, k: usize) -> anyhow::Result<Vec<SearchHit>> {
        let query = query.trim();
        if query.is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        let pool = k.saturating_mul(4).max(1);
        let t = Instant::now();
        let lexical = self.bm25.search(query, pool);
        let dense = match self.embedder.as_mut() {
            Some(e) if !self.vectors.is_empty() => {
                let q = e.embed_query(query)?;
                self.vectors.search(&q, pool)
            }
            _ => Vec::new(),
        };
        let mut fused = hybrid::fuse(&lexical, &dense, Weights::default(), pool);
        let needle = query.to_lowercase();
        for f in &mut fused {
            if self.chunks[f.chunk_id]
                .text
                .to_lowercase()
                .contains(&needle)
            {
                f.score += PHRASE_BONUS;
            }
        }
        fused.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then(a.chunk_id.cmp(&b.chunk_id))
        });
        fused.truncate(k);
        tracing::info!(
            lexical = lexical.len(),
            dense = dense.len(),
            ms = t.elapsed().as_millis(),
            "searched"
        );
        Ok(fused
            .into_iter()
            .map(|f| {
                let c = &self.chunks[f.chunk_id];
                SearchHit {
                    chunk_id: c.id,
                    path: c.path.display().to_string(),
                    heading_path: c.heading_path.clone(),
                    line_start: c.line_start,
                    line_end: c.line_end,
                    score: f.score,
                    snippet: snippet(&c.text),
                }
            })
            .collect())
    }

    pub fn get_chunk(&self, id: usize) -> Option<&Chunk> {
        self.chunks.get(id)
    }

    /// Full text of every chunk from `path`, in order, joined with blank lines.
    pub fn get_document(&self, path: &Path) -> Option<String> {
        let parts: Vec<&str> = self
            .chunks
            .iter()
            .filter(|c| c.path == path)
            .map(|c| c.text.as_str())
            .collect();
        if parts.is_empty() {
            None
        } else {
            Some(parts.join("\n\n"))
        }
    }

    pub fn stats(&self) -> serde_json::Value {
        let files: std::collections::HashSet<&Path> =
            self.chunks.iter().map(|c| c.path.as_path()).collect();
        serde_json::json!({
            "files": files.len(),
            "chunks": self.chunks.len(),
            "lexical_terms": self.bm25.term_count(),
            "dense": self.embedder.is_some(),
            "dim": self.embedder.as_ref().map_or(0, Embedder::dim),
        })
    }
}

fn snippet(text: &str) -> String {
    match text.char_indices().nth(SNIPPET_CHARS) {
        Some((i, _)) => format!("{}…", &text[..i]),
        None => text.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snippet_is_char_boundary_safe() {
        let s = "é".repeat(SNIPPET_CHARS + 5);
        let out = snippet(&s);
        assert_eq!(out.chars().count(), SNIPPET_CHARS + 1);
        assert!(out.ends_with('…'));
        assert_eq!(snippet("short"), "short");
    }
}
