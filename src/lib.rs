//! oasis — in-memory hybrid search over a markdown corpus.
//!
//! Pipeline: [`corpus`] walks + chunks markdown → [`bm25`] builds a lexical
//! index and [`embed`]+[`vector`] build a dense index → [`hybrid`] fuses the
//! two rankings → the CLI exposes `search` (with `--json` for AI callers).
//! [`cache`] persists embeddings across invocations so `search` stays fast.

pub mod bm25;
pub mod cache;
pub mod config;
pub mod corpus;
pub mod embed;
pub mod engine;
pub mod hybrid;
pub mod vector;
