# oasis

In-memory hybrid search over a personal corpus of markdown docs and references,
built as a CLI tool that AI models can call for relevant context.

- **Lexical**: hand-rolled BM25 with identifier-aware tokenization (`tokio::spawn`
  matches `tokio`, `spawn`, and the whole token) and heading boosting.
- **Dense**: `BAAI/bge-small-en-v1.5` in ONNX via [`ort`](https://ort.pyke.io),
  flat cosine index, embeddings cached on disk so repeat calls only embed changes.
- **Fusion**: reciprocal rank fusion + exact-phrase bonus.
- **Chunking**: heading-aware, never splits a fenced code block.

## Usage

```sh
oasis --root ~/notes --root ~/refs index                 # build + print stats
oasis --root ~/notes search "graceful shutdown tokio"    # human-readable
oasis --root ~/notes search "graceful shutdown tokio" --json -k 5
oasis --root ~/notes show --chunk 42                     # full chunk by id
oasis --root ~/notes show --path rust/tokio.md           # whole document
oasis --root ~/notes --lexical-only search "..."         # BM25 only, no model
```

`OASIS_ROOTS` (colon-separated) replaces `--root`. `OASIS_MODEL_DIR` points at a
local `model.onnx` + `tokenizer.json`; otherwise the model is fetched from the
Hugging Face hub on first use. The embedding cache lives in `$OASIS_CACHE_DIR`
or `~/.cache/oasis/`.

## Layout

| module      | role                                                    |
|-------------|---------------------------------------------------------|
| `corpus`    | walk roots, chunk markdown by heading                   |
| `bm25`      | tokenizer + inverted index + BM25 scoring               |
| `embed`     | ONNX session, tokenizer, CLS pooling, normalization     |
| `vector`    | flat cosine index                                       |
| `cache`     | on-disk embedding cache keyed by blake3(chunk text)     |
| `hybrid`    | reciprocal rank fusion                                  |
| `engine`    | owns corpus + indexes, answers queries                  |
| `main`      | clap CLI: `index`, `search`, `show`                     |
