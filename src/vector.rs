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
        Self {
            dim,
            data: Vec::new(),
        }
    }

    pub fn push(&mut self, v: &[f32]) {
        assert_eq!(
            v.len(),
            self.dim,
            "vector dim mismatch: expected {}, got {}",
            self.dim,
            v.len()
        );
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            self.data.extend(v.iter().map(|x| x / norm));
        } else {
            // Zero vector: store as-is rather than dividing by zero.
            self.data.extend_from_slice(v);
        }
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let n = self.len();
        let mut scores: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                let row = &self.data[i * self.dim..(i + 1) * self.dim];
                let dot: f32 = row.iter().zip(query.iter()).map(|(a, b)| a * b).sum();
                (i, dot)
            })
            .collect();
        // Top-k via partial sort, then finish sorting the kept prefix desc.
        if k < scores.len() {
            scores.select_nth_unstable_by(k, |a, b| b.1.total_cmp(&a.1));
            scores.truncate(k);
        }
        scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        scores
    }

    pub fn len(&self) -> usize {
        self.data.len().checked_div(self.dim).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_normalizes() {
        let mut idx = VectorIndex::new(2);
        idx.push(&[3.0, 4.0]);
        assert_eq!(idx.len(), 1);
        let hits = idx.search(&[0.6, 0.8], 1);
        assert!((hits[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn top_k_sorted_desc() {
        let mut idx = VectorIndex::new(2);
        idx.push(&[1.0, 0.0]);
        idx.push(&[0.0, 1.0]);
        idx.push(&[1.0, 1.0]);
        let hits = idx.search(&[1.0, 0.0], 2);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].0, 0);
        assert_eq!(hits[1].0, 2);
        assert!(hits[0].1 >= hits[1].1);
    }

    #[test]
    fn zero_vector_is_stored_and_scores_zero() {
        let mut idx = VectorIndex::new(2);
        idx.push(&[0.0, 0.0]);
        let hits = idx.search(&[1.0, 0.0], 1);
        assert_eq!(hits, vec![(0, 0.0)]);
    }

    #[test]
    #[should_panic]
    fn wrong_dim_panics() {
        let mut idx = VectorIndex::new(2);
        idx.push(&[1.0]);
    }
}
