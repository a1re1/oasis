//! Rank fusion between lexical and dense results.
//!
//! Contract:
//! - Reciprocal Rank Fusion: `score(d) = Σ_i w_i / (RRF_K + rank_i(d))` over
//!   the lists the doc appears in; `RRF_K = 60`; default weights
//!   `lexical = 1.0`, `dense = 1.0`.
//! - Ties broken by dense score, then chunk id.
//! - `fuse` takes each list already sorted desc and returns fused results
//!   sorted desc, truncated to `k`.
//! - Exact-phrase bonus: if the raw query (case-insensitive) appears verbatim
//!   in a chunk's text, add `PHRASE_BONUS` (0.02) to its fused score.

pub const RRF_K: f32 = 60.0;
pub const PHRASE_BONUS: f32 = 0.02;

#[derive(Debug, Clone, Copy)]
pub struct Weights {
    pub lexical: f32,
    pub dense: f32,
}

impl Default for Weights {
    fn default() -> Self {
        Self { lexical: 1.0, dense: 1.0 }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Fused {
    pub chunk_id: usize,
    pub score: f32,
    pub lexical_rank: Option<usize>,
    pub dense_rank: Option<usize>,
}

pub fn fuse(lexical: &[(usize, f32)], dense: &[(usize, f32)], w: Weights, k: usize) -> Vec<Fused> {
    let _ = (lexical, dense, w, k);
    todo!()
}
