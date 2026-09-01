//! On-disk embedding cache so repeated CLI invocations don't re-embed an
//! unchanged corpus. The engine itself is still fully in-memory; this only
//! short-circuits the expensive model calls.
//!
//! Contract:
//! - Location: `$OASIS_CACHE_DIR` or `dirs::cache_dir()/oasis/<model_repo slug>.bin`.
//!   One file per model, so switching models never mixes dimensions.
//! - Key: `blake3(indexable_text)` (32 bytes). Value: `Vec<f32>` of `dim`.
//! - `load(cfg)` returns an empty cache if the file is missing or unreadable
//!   (log a warning, never fail). `save()` writes atomically (tmp + rename).
//! - `get_or_embed(embedder, texts) -> Vec<Vec<f32>>` looks up every text,
//!   embeds only the misses in one batched call, inserts them, and returns
//!   vectors in input order. Reports hit/miss counts via `tracing::info!`.
//! - Serialization via `bincode`.

use std::collections::HashMap;
use std::path::PathBuf;

use crate::{config::Config, embed::Embedder};

#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct EmbeddingCache {
    pub dim: usize,
    pub entries: HashMap<[u8; 32], Vec<f32>>,
    #[serde(skip)]
    path: Option<PathBuf>,
    #[serde(skip)]
    dirty: bool,
}

impl EmbeddingCache {
    pub fn cache_path(cfg: &Config) -> PathBuf {
        let _ = cfg;
        todo!()
    }

    pub fn load(cfg: &Config) -> Self {
        let _ = cfg;
        todo!()
    }

    pub fn save(&self) -> anyhow::Result<()> {
        todo!()
    }

    pub fn get_or_embed(&mut self, embedder: &mut Embedder, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        let _ = (embedder, texts);
        todo!()
    }
}
