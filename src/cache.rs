//! On-disk embedding cache so repeated CLI invocations don't re-embed an
//! unchanged corpus. The engine itself is still fully in-memory; this only
//! short-circuits the expensive model calls.
//!
//! Contract:
//! - Location: `cfg.cache_dir`, else `$OASIS_CACHE_DIR`, else
//!   `dirs::cache_dir()/oasis`; file name `<model_repo slug>.bin`.
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

fn key(text: &str) -> [u8; 32] {
    *blake3::hash(text.as_bytes()).as_bytes()
}

impl EmbeddingCache {
    pub fn cache_path(cfg: &Config) -> PathBuf {
        let dir = cfg
            .cache_dir
            .clone()
            .or_else(|| std::env::var_os("OASIS_CACHE_DIR").map(PathBuf::from))
            .unwrap_or_else(|| {
                dirs::cache_dir()
                    .unwrap_or_else(std::env::temp_dir)
                    .join("oasis")
            });
        let slug = cfg.model_repo.replace('/', "__");
        dir.join(format!("{slug}.bin"))
    }

    pub fn load(cfg: &Config) -> Self {
        let path = Self::cache_path(cfg);
        let mut cache = match std::fs::read(&path) {
            Ok(bytes) => match bincode::deserialize::<EmbeddingCache>(&bytes) {
                Ok(c) if !c.is_consistent() => {
                    tracing::warn!(path = %path.display(), "embedding cache has mixed dimensions or non-finite values; starting empty");
                    Self::default()
                }
                Ok(c) => {
                    tracing::info!(entries = c.entries.len(), path = %path.display(), "loaded embedding cache");
                    c
                }
                Err(e) => {
                    tracing::warn!(path = %path.display(), error = %e, "corrupt embedding cache; starting empty");
                    Self::default()
                }
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Self::default(),
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "unreadable embedding cache; starting empty");
                Self::default()
            }
        };
        cache.path = Some(path);
        cache.dirty = false;
        cache
    }

    pub fn save(&self) -> anyhow::Result<()> {
        if !self.dirty {
            return Ok(());
        }
        let Some(path) = &self.path else {
            return Ok(());
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("bin.tmp");
        std::fs::write(&tmp, bincode::serialize(self)?)?;
        std::fs::rename(&tmp, path)?;
        tracing::info!(entries = self.entries.len(), path = %path.display(), "saved embedding cache");
        Ok(())
    }

    /// True when every entry has `dim` finite components.
    fn is_consistent(&self) -> bool {
        self.entries
            .values()
            .all(|v| v.len() == self.dim && v.iter().all(|x| x.is_finite()))
    }

    /// Insert a precomputed vector. Exposed so the lookup/insert path can be
    /// tested without a model.
    pub fn insert(&mut self, text: &str, v: Vec<f32>) {
        self.dim = v.len();
        self.entries.insert(key(text), v);
        self.dirty = true;
    }

    pub fn get(&self, text: &str) -> Option<&Vec<f32>> {
        self.entries.get(&key(text))
    }

    pub fn get_or_embed(
        &mut self,
        embedder: &mut Embedder,
        texts: &[&str],
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        let keys: Vec<[u8; 32]> = texts.iter().map(|t| key(t)).collect();
        let misses: Vec<usize> = (0..texts.len())
            .filter(|&i| !self.entries.contains_key(&keys[i]))
            .collect();
        tracing::info!(
            hits = texts.len() - misses.len(),
            misses = misses.len(),
            "embedding cache"
        );
        if !misses.is_empty() {
            let miss_texts: Vec<&str> = misses.iter().map(|&i| texts[i]).collect();
            let vecs = embedder.embed(&miss_texts)?;
            anyhow::ensure!(vecs.len() == misses.len(), "embedder returned wrong count");
            for (&i, v) in misses.iter().zip(vecs) {
                self.dim = v.len();
                self.entries.insert(keys[i], v);
            }
            self.dirty = true;
        }
        Ok(keys.iter().map(|k| self.entries[k].clone()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_in(dir: &std::path::Path) -> Config {
        Config {
            cache_dir: Some(dir.to_path_buf()),
            ..Config::default()
        }
    }

    #[test]
    fn roundtrip_save_load() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = cfg_in(dir.path());
        let mut c = EmbeddingCache::load(&cfg);
        assert!(c.entries.is_empty());
        c.insert("hello", vec![0.0, 1.0]);
        c.save().unwrap();
        let c2 = EmbeddingCache::load(&cfg);
        assert_eq!(c2.get("hello"), Some(&vec![0.0, 1.0]));
        assert_eq!(c2.get("other"), None);
        assert_eq!(c2.dim, 2);
        assert!(EmbeddingCache::cache_path(&cfg).ends_with("BAAI__bge-small-en-v1.5.bin"));
    }

    #[test]
    fn inconsistent_dims_start_empty() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = cfg_in(dir.path());
        let mut c = EmbeddingCache::load(&cfg);
        c.insert("a", vec![1.0, 2.0]);
        c.entries.insert([7u8; 32], vec![1.0]);
        c.save().unwrap();
        assert!(EmbeddingCache::load(&cfg).entries.is_empty());
    }

    #[test]
    fn corrupt_file_starts_empty() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = cfg_in(dir.path());
        std::fs::write(EmbeddingCache::cache_path(&cfg), b"garbage").unwrap();
        let c = EmbeddingCache::load(&cfg);
        assert!(c.entries.is_empty());
    }
}
