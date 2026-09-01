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

use rust_stemmers::{Algorithm, Stemmer};
use unicode_segmentation::UnicodeSegmentation;

use crate::corpus::Chunk;

pub const K1: f32 = 1.2;
pub const B: f32 = 0.75;
pub const HEADING_BOOST: f32 = 2.0;

/// Separator characters that split an identifier into its parts.
const IDENT_SEPARATORS: [char; 4] = ['_', '.', ':', '-'];

fn stem_word(stemmer: &Stemmer, word: &str) -> String {
    stemmer.stem(word).into_owned()
}

/// Split an identifier-shaped word into its alphabetic/numeric parts.
fn identifier_parts(word: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    for ch in word.chars() {
        if IDENT_SEPARATORS.contains(&ch) {
            if !current.is_empty() {
                parts.push(std::mem::take(&mut current));
            }
        } else {
            current.push(ch);
        }
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts
}

fn is_separator_run(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| IDENT_SEPARATORS.contains(&c))
}

/// A word is identifier-shaped when a separator occurs between alphanumeric
/// characters (e.g. `tokio::spawn`, `my_var`, `foo.bar`), not merely at the edges.
fn has_internal_separator(word: &str) -> bool {
    let mut prev_alnum = false;
    let mut saw_sep_after_alnum = false;
    for ch in word.chars() {
        if IDENT_SEPARATORS.contains(&ch) {
            if prev_alnum {
                saw_sep_after_alnum = true;
            }
            prev_alnum = false;
        } else if ch.is_alphanumeric() {
            if saw_sep_after_alnum {
                return true;
            }
            prev_alnum = true;
        }
    }
    false
}

fn all_alphabetic(word: &str) -> bool {
    !word.is_empty() && word.chars().all(|c| c.is_alphabetic())
}

/// Push a token, stemming it first when purely alphabetic.
fn emit(out: &mut Vec<String>, stemmer: &Stemmer, token: &str) {
    if all_alphabetic(token) {
        out.push(stem_word(stemmer, token));
    } else {
        out.push(token.to_string());
    }
}

/// Reassemble `split_word_bounds` segments into identifier-aware words:
/// a separator-only segment glues the alphanumeric words on either side.
fn words(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut pending_sep = false;
    for segment in text.split_word_bounds() {
        let is_sep = is_separator_run(segment);
        let has_alnum = segment.chars().any(|c| c.is_alphanumeric());
        if is_sep && !current.is_empty() {
            current.push_str(segment);
            pending_sep = true;
        } else if has_alnum {
            if pending_sep {
                pending_sep = false;
                current.push_str(segment);
            } else {
                if !current.is_empty() {
                    out.push(std::mem::take(&mut current));
                }
                current.push_str(segment);
            }
        } else if !current.is_empty() {
            out.push(std::mem::take(&mut current));
            pending_sep = false;
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    // A separator run at end-of-text (`my_var.`) is punctuation, not identifier.
    for tok in &mut out {
        let trimmed = tok.trim_end_matches(|c: char| !c.is_alphanumeric());
        if trimmed.len() != tok.len() {
            *tok = trimmed.to_string();
        }
    }
    out.retain(|t| !t.is_empty());
    out
}

pub fn tokenize(text: &str) -> Vec<String> {
    let stemmer = Stemmer::create(Algorithm::English);
    let lowered = text.to_lowercase();
    let mut out = Vec::new();
    for word in words(&lowered) {
        if has_internal_separator(&word) {
            emit(&mut out, &stemmer, &word);
            for part in identifier_parts(&word) {
                if !part.is_empty() {
                    emit(&mut out, &stemmer, &part);
                }
            }
        } else {
            emit(&mut out, &stemmer, &word);
        }
    }
    out
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

/// Term frequencies for one chunk: heading tokens count `HEADING_BOOST` times.
fn term_freqs(chunks: &[Chunk], id: usize) -> (HashMap<String, f32>, f32) {
    let mut tf: HashMap<String, f32> = HashMap::new();
    let mut total = 0.0f32;
    let chunk = &chunks[id];
    let add = |text: &str, weight: f32, tf: &mut HashMap<String, f32>, total: &mut f32| {
        for tok in tokenize(text) {
            *tf.entry(tok).or_insert(0.0) += weight;
            *total += weight;
        }
    };
    for heading in &chunk.heading_path {
        add(heading, HEADING_BOOST, &mut tf, &mut total);
    }
    // Body only: `indexable_text()` would re-add the heading path at 1.0.
    add(&chunk.text, 1.0, &mut tf, &mut total);
    (tf, total)
}

impl Bm25Index {
    pub fn build(chunks: &[Chunk]) -> Self {
        let mut index = Bm25Index {
            doc_len: Vec::with_capacity(chunks.len()),
            n_docs: chunks.len(),
            ..Bm25Index::default()
        };
        for chunk in chunks {
            let (tf, len) = term_freqs(chunks, chunk.id);
            index.doc_len.push(len);
            for (term, freq) in tf {
                index
                    .postings
                    .entry(term)
                    .or_default()
                    .push((chunk.id, freq));
            }
        }
        index.avg_doc_len = if index.doc_len.is_empty() {
            0.0
        } else {
            index.doc_len.iter().sum::<f32>() / index.doc_len.len() as f32
        };
        index
    }

    pub fn search(&self, query: &str, k: usize) -> Vec<(usize, f32)> {
        let mut scores: HashMap<usize, f32> = HashMap::new();
        for term in tokenize(query) {
            let Some(postings) = self.postings.get(&term) else {
                continue;
            };
            let n = postings.len() as f32;
            let idf = (1.0 + (self.n_docs as f32 - n + 0.5) / (n + 0.5)).ln();
            for &(id, tf) in postings {
                let dl = self.doc_len.get(id).copied().unwrap_or(0.0);
                let norm = if self.avg_doc_len > 0.0 {
                    dl / self.avg_doc_len
                } else {
                    0.0
                };
                let denom = tf + K1 * (1.0 - B + B * norm);
                let score = idf * tf * (K1 + 1.0) / denom;
                *scores.entry(id).or_insert(0.0) += score;
            }
        }
        let mut hits: Vec<(usize, f32)> = scores.into_iter().collect();
        hits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        hits.truncate(k);
        hits
    }

    pub fn len(&self) -> usize {
        self.n_docs
    }

    pub fn is_empty(&self) -> bool {
        self.n_docs == 0
    }

    /// Number of distinct terms in the index.
    pub fn term_count(&self) -> usize {
        self.postings.len()
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    fn chunk(id: usize, heading: &[&str], text: &str) -> Chunk {
        Chunk {
            id,
            path: Path::new("t.md").to_path_buf(),
            root: Path::new("/r").to_path_buf(),
            heading_path: heading.iter().map(|s| s.to_string()).collect(),
            text: text.to_string(),
            line_start: 1,
            line_end: 2,
        }
    }

    #[test]
    fn trailing_separator_is_not_part_of_identifier() {
        let toks = tokenize("call my_var.");
        assert!(toks.contains(&"my_var".to_string()), "{toks:?}");
        assert!(!toks.iter().any(|t| t.ends_with('.')), "{toks:?}");
        assert!(tokenize("...").is_empty());
    }

    #[test]
    fn tokenize_emits_identifier_and_parts() {
        let toks = tokenize("tokio::spawn");
        assert!(toks.contains(&"tokio::spawn".to_string()), "{toks:?}");
        assert!(toks.contains(&"tokio".to_string()), "{toks:?}");
        assert!(toks.contains(&"spawn".to_string()), "{toks:?}");
    }

    #[test]
    fn tokenize_stems_english() {
        let toks = tokenize("running quickly indexed");
        assert_eq!(toks, vec!["run", "quick", "index"]);
    }

    #[test]
    fn more_matches_rank_higher() {
        let a = chunk(0, &["Tokio"], "spawn a task with spawn");
        let b = chunk(1, &["Other"], "spawn once");
        let idx = Bm25Index::build(&[a, b]);
        let hits = idx.search("tokio spawn task", 2);
        assert_eq!(hits[0].0, 0);
    }

    #[test]
    fn heading_hit_ranks_higher() {
        let a = chunk(0, &["Runtime"], "unrelated body text here");
        let b = chunk(1, &["Other"], "runtime in body only");
        let idx = Bm25Index::build(&[a, b]);
        let hits = idx.search("runtime", 2);
        assert_eq!(hits[0].0, 0);
    }
}
