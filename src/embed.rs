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

use std::path::PathBuf;

use anyhow::Context;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::config::Config;

pub const QUERY_INSTRUCTION: &str = "Represent this sentence for searching relevant passages: ";
pub const MAX_TOKENS: usize = 512;
pub const BATCH_SIZE: usize = 32;

pub struct Embedder {
    session: Session,
    tokenizer: Tokenizer,
    dim: usize,
    output_name: String,
}

fn resolve_files(cfg: &Config) -> anyhow::Result<(PathBuf, PathBuf)> {
    if let Some(dir) = &cfg.model_dir {
        let model = dir.join("model.onnx");
        let tok = dir.join("tokenizer.json");
        anyhow::ensure!(model.is_file(), "missing {}", model.display());
        anyhow::ensure!(tok.is_file(), "missing {}", tok.display());
        return Ok((model, tok));
    }
    let (owner, name) = cfg
        .model_repo
        .split_once('/')
        .with_context(|| format!("model repo must be owner/name, got {}", cfg.model_repo))?;
    let client = hf_hub::HFClientSync::new().context("init hf-hub")?;
    let repo = client.model(owner, name);
    let fetch = |file: &str| {
        repo.download_file()
            .filename(file)
            .send()
            .with_context(|| format!("fetch {file} from {}", cfg.model_repo))
    };
    Ok((fetch("onnx/model.onnx")?, fetch("tokenizer.json")?))
}

impl Embedder {
    pub fn load(cfg: &Config) -> anyhow::Result<Self> {
        let (model_path, tok_path) = resolve_files(cfg)?;
        tracing::info!(model = %model_path.display(), "loading embedding model");
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let ort_err =
            |e: &dyn std::fmt::Display| anyhow::anyhow!("load {}: {e}", model_path.display());
        let mut builder = Session::builder()
            .map_err(|e| ort_err(&e))?
            .with_intra_threads(threads)
            .map_err(|e| ort_err(&e))?;
        let session = builder
            .commit_from_file(&model_path)
            .map_err(|e| ort_err(&e))?;
        let output_name = session
            .outputs()
            .first()
            .map(|o| o.name().to_string())
            .context("model has no outputs")?;
        let mut tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("load tokenizer {}: {e}", tok_path.display()))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_TOKENS,
                ..Default::default()
            }))
            .map_err(|e| anyhow::anyhow!("set truncation: {e}"))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..Default::default()
        }));
        Ok(Self {
            session,
            tokenizer,
            dim: cfg_dim_hint(cfg),
            output_name,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn embed(&mut self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for batch in texts.chunks(BATCH_SIZE) {
            out.extend(self.embed_batch(batch)?);
        }
        Ok(out)
    }

    pub fn embed_query(&mut self, query: &str) -> anyhow::Result<Vec<f32>> {
        let text = format!("{QUERY_INSTRUCTION}{query}");
        let mut v = self.embed_batch(&[&text])?;
        v.pop().context("empty embedding batch")
    }

    fn embed_batch(&mut self, batch: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }
        let encodings = self
            .tokenizer
            .encode_batch(batch.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        let b = encodings.len();
        let s = encodings.first().map_or(0, |e| e.get_ids().len());
        let mut ids = Vec::with_capacity(b * s);
        let mut mask = Vec::with_capacity(b * s);
        let mut types = Vec::with_capacity(b * s);
        for e in &encodings {
            ids.extend(e.get_ids().iter().map(|&x| x as i64));
            mask.extend(e.get_attention_mask().iter().map(|&x| x as i64));
            types.extend(e.get_type_ids().iter().map(|&x| x as i64));
        }
        let shape = vec![b as i64, s as i64];
        let outputs = self.session.run(ort::inputs! {
            "input_ids" => Tensor::from_array((shape.clone(), ids))?,
            "attention_mask" => Tensor::from_array((shape.clone(), mask))?,
            "token_type_ids" => Tensor::from_array((shape, types))?,
        })?;
        let hidden = outputs[self.output_name.as_str()].try_extract_array::<f32>()?;
        let shape = hidden.shape();
        anyhow::ensure!(
            shape.len() == 3,
            "expected [batch, seq, dim], got {shape:?}"
        );
        let dim = shape[2];
        self.dim = dim;
        let mut vecs = Vec::with_capacity(b);
        for i in 0..b {
            let mut v: Vec<f32> = (0..dim).map(|d| hidden[[i, 0, d]]).collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                v.iter_mut().for_each(|x| *x /= norm);
            }
            vecs.push(v);
        }
        Ok(vecs)
    }
}

/// Until the first batch runs we only know the configured default dimension.
fn cfg_dim_hint(_cfg: &Config) -> usize {
    crate::config::DEFAULT_EMBED_DIM
}
