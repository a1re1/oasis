//! Flat (brute-force) cosine index. Vectors are L2-normalized on insert so
//! cosine == dot product. Fine up to ~10^5 chunks; swap for HNSW later.
//!
//! Contract: `search(query, k)` returns at most `k` `(chunk_id, score)` sorted
//! desc, where `chunk_id` is the insertion index.

#[derive(Debug, Default)]
pub struct VectorIndex {
    dim: usize,
    /// Row-major, `len * dim`.
    data: Vec<f32>,
}

impl VectorIndex {
    pub fn new(dim: usize) -> Self {
        Self { dim, data: Vec::new() }
    }

    pub fn push(&mut self, v: &[f32]) {
        let _ = v;
        todo!("normalize + append; panic if v.len() != dim")
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let _ = (query, k);
        todo!()
    }

    pub fn len(&self) -> usize {
        if self.dim == 0 { 0 } else { self.data.len() / self.dim }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
