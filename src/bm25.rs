//! In-memory BM25 inverted index over chunks.
//!
//! Contract:
//! - `tokenize(text)` lowercases, splits on Unicode word boundaries
//!   (`unicode-segmentation`), drops pure punctuation, keeps tokens that
//!   contain `_`, `.`, `::`, `-` inside identifiers as a single token AND also
//!   emits their split parts (so `tokio::spawn` matches `tokio`, `spawn`, and
//!   `tokio::spawn`). Applies English Snowball stemming to alphabetic tokens.
//! - Standard BM25 with `k1 = 1.2`, `b = 0.75`, IDF = ln(1 + (N - n + 0.5)/(n + 0.5)).
//! - Heading tokens are indexed with a `heading_boost` multiplier on term
//!   frequency (default 2.0) so a query hitting a section title ranks higher.
//! - `search(query, k)` returns at most `k` `(chunk_id, score)` sorted desc.

use std::collections::HashMap;

use crate::corpus::Chunk;

pub const K1: f32 = 1.2;
pub const B: f32 = 0.75;
pub const HEADING_BOOST: f32 = 2.0;

pub fn tokenize(text: &str) -> Vec<String> {
    let _ = text;
    todo!()
}

#[derive(Debug, Default)]
pub struct Bm25Index {
    /// term -> postings of (chunk_id, weighted term frequency)
    postings: HashMap<String, Vec<(usize, f32)>>,
    /// per-chunk weighted length
    doc_len: Vec<f32>,
    avg_doc_len: f32,
    n_docs: usize,
}

impl Bm25Index {
    pub fn build(chunks: &[Chunk]) -> Self {
        let _ = chunks;
        todo!()
    }

    pub fn search(&self, query: &str, k: usize) -> Vec<(usize, f32)> {
        let _ = (query, k);
        todo!()
    }

    pub fn len(&self) -> usize {
        self.n_docs
    }

    pub fn is_empty(&self) -> bool {
        self.n_docs == 0
    }
}
