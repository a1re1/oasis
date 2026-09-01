//! ONNX embedding model via `ort`.
//!
//! Contract:
//! - `Embedder::load(cfg)` resolves the model: if `cfg.model_dir` is set, read
//!   `model.onnx` + `tokenizer.json` from it; otherwise fetch
//!   `onnx/model.onnx` and `tokenizer.json` from `cfg.model_repo` via `hf-hub`
//!   (cached in the default HF cache dir).
//! - `embed(&[&str]) -> Vec<Vec<f32>>` batches inputs (batch size 32),
//!   truncates to 512 tokens, feeds `input_ids` / `attention_mask` /
//!   `token_type_ids` (i64), takes the CLS token (index 0) of
//!   `last_hidden_state` (bge convention) and L2-normalizes each vector.
//! - `embed_query(q)` prefixes bge's recommended instruction
//!   `"Represent this sentence for searching relevant passages: "`.
//! - `dim()` reports the output dimension (384 for bge-small).

use crate::config::Config;

pub const QUERY_INSTRUCTION: &str = "Represent this sentence for searching relevant passages: ";
pub const MAX_TOKENS: usize = 512;
pub const BATCH_SIZE: usize = 32;

pub struct Embedder {
    session: ort::session::Session,
    tokenizer: tokenizers::Tokenizer,
    dim: usize,
}

impl Embedder {
    pub fn load(cfg: &Config) -> anyhow::Result<Self> {
        let _ = cfg;
        todo!()
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn embed(&mut self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        let _ = texts;
        todo!()
    }

    pub fn embed_query(&mut self, query: &str) -> anyhow::Result<Vec<f32>> {
        let _ = query;
        todo!()
    }
}
