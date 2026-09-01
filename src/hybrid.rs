//! Rank fusion between lexical and dense results.
//!
//! Contract:
//! - Reciprocal Rank Fusion: `score(d) = Σ_i w_i / (RRF_K + rank_i(d))` over
//!   the lists the doc appears in; `RRF_K = 60`; default weights
//!   `lexical = 1.0`, `dense = 1.0`. Ranks are 1-based.
//! - Ties broken by dense score, then chunk id.
//! - `fuse` takes each list already sorted desc and returns fused results
//!   sorted desc, truncated to `k`.
//! - Exact-phrase bonus: if the raw query (case-insensitive) appears verbatim
//!   in a chunk's text, the engine adds `PHRASE_BONUS` (0.02) to its fused
//!   score. Applied in `engine`, which owns the chunk text.

use std::collections::HashMap;

pub const RRF_K: f32 = 60.0;
pub const PHRASE_BONUS: f32 = 0.02;

#[derive(Debug, Clone, Copy)]
pub struct Weights {
    pub lexical: f32,
    pub dense: f32,
}

impl Default for Weights {
    fn default() -> Self {
        Self {
            lexical: 1.0,
            dense: 1.0,
        }
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
    struct Acc {
        score: f32,
        lexical_rank: Option<usize>,
        dense_rank: Option<usize>,
        dense_score: f32,
    }
    fn entry(acc: &mut HashMap<usize, Acc>, id: usize) -> &mut Acc {
        acc.entry(id).or_insert(Acc {
            score: 0.0,
            lexical_rank: None,
            dense_rank: None,
            dense_score: f32::NEG_INFINITY,
        })
    }
    let mut acc: HashMap<usize, Acc> = HashMap::new();
    for (rank, (id, _)) in lexical.iter().enumerate() {
        let a = entry(&mut acc, *id);
        a.score += w.lexical / (RRF_K + (rank + 1) as f32);
        a.lexical_rank = Some(rank + 1);
    }
    for (rank, (id, s)) in dense.iter().enumerate() {
        let a = entry(&mut acc, *id);
        a.score += w.dense / (RRF_K + (rank + 1) as f32);
        a.dense_rank = Some(rank + 1);
        a.dense_score = *s;
    }
    let mut out: Vec<(Fused, f32)> = acc
        .into_iter()
        .map(|(chunk_id, a)| {
            (
                Fused {
                    chunk_id,
                    score: a.score,
                    lexical_rank: a.lexical_rank,
                    dense_rank: a.dense_rank,
                },
                a.dense_score,
            )
        })
        .collect();
    out.sort_by(|(a, ad), (b, bd)| {
        b.score
            .total_cmp(&a.score)
            .then(bd.total_cmp(ad))
            .then(a.chunk_id.cmp(&b.chunk_id))
    });
    out.truncate(k);
    out.into_iter().map(|(f, _)| f).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doc_in_both_lists_beats_doc_in_one() {
        let lex = [(1, 5.0), (2, 4.0)];
        let den = [(3, 0.9), (2, 0.8)];
        let out = fuse(&lex, &den, Weights::default(), 10);
        assert_eq!(out[0].chunk_id, 2);
        assert_eq!(out[0].lexical_rank, Some(2));
        assert_eq!(out[0].dense_rank, Some(2));
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn truncates_to_k() {
        let lex = [(1, 1.0), (2, 1.0), (3, 1.0)];
        let out = fuse(&lex, &[], Weights::default(), 2);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].chunk_id, 1);
    }

    #[test]
    fn weights_shift_ranking() {
        let lex = [(1, 1.0)];
        let den = [(2, 1.0)];
        let w = Weights {
            lexical: 0.1,
            dense: 1.0,
        };
        let out = fuse(&lex, &den, w, 10);
        assert_eq!(out[0].chunk_id, 2);
    }

    #[test]
    fn tie_breaks_on_dense_then_id() {
        let den = [(7, 0.5), (3, 0.5)];
        let lex = [(3, 1.0), (7, 1.0)];
        // Both have rank 1 in one list and rank 2 in the other -> equal score;
        // equal dense score -> lower id wins.
        let out = fuse(&lex, &den, Weights::default(), 10);
        assert_eq!(out[0].chunk_id, 3);
    }
}
