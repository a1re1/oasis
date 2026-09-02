use std::path::PathBuf;

use clap::{Parser, Subcommand};
use oasis::config::Config;

#[derive(Parser)]
#[command(
    name = "oasis",
    version,
    about = "Hybrid (dense + BM25) search over your markdown corpus. Use --json for machine consumption."
)]
struct Cli {
    /// Directories to ingest. Repeatable. Defaults to $OASIS_ROOTS (colon-separated).
    #[arg(
        short,
        long = "root",
        env = "OASIS_ROOTS",
        value_delimiter = ':',
        global = true
    )]
    roots: Vec<PathBuf>,

    /// HF repo id of the ONNX embedding model.
    #[arg(long, env = "OASIS_MODEL", default_value = oasis::config::DEFAULT_MODEL_REPO, global = true)]
    model: String,

    /// Local dir with model.onnx + tokenizer.json (overrides --model).
    #[arg(long, env = "OASIS_MODEL_DIR", global = true)]
    model_dir: Option<PathBuf>,

    /// Disable dense (ONNX) search; BM25 only.
    #[arg(long, global = true)]
    lexical_only: bool,

    /// Directory for the on-disk embedding cache (default: platform cache dir).
    #[arg(long, env = "OASIS_CACHE_DIR", global = true)]
    cache_dir: Option<PathBuf>,

    /// Glob patterns for files to exclude from indexing (path relative to the
    /// root, e.g. "index.md", "log/**"). Repeatable; also $OASIS_IGNORE
    /// (colon-separated).
    #[arg(
        long = "ignore",
        value_name = "GLOB",
        env = "OASIS_IGNORE",
        value_delimiter = ':',
        global = true
    )]
    ignore: Vec<String>,

    /// Maximum hits per document path in search results
    /// (1 = one hit per document, 0 = every matching chunk).
    #[arg(long, value_name = "N", default_value_t = 1, global = true)]
    per_page: usize,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Build the index and print stats (smoke test for ingestion).
    Index,
    /// Build the index and run one query from the CLI.
    Search {
        query: String,
        #[arg(short = 'k', long, default_value_t = 8)]
        top_k: usize,
        /// Print results as JSON instead of a human-readable list.
        #[arg(long)]
        json: bool,
    },
    /// Print the full text of one chunk (by id from a previous search) or a whole document (by path).
    Show {
        /// Chunk id from `search --json`.
        #[arg(long, conflicts_with = "path")]
        chunk: Option<usize>,
        /// Relative path as printed by `search`.
        #[arg(long)]
        path: Option<PathBuf>,
    },
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("oasis=info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    let cfg = Config {
        roots: cli.roots,
        model_repo: cli.model,
        model_dir: cli.model_dir,
        lexical_only: cli.lexical_only,
        cache_dir: cli.cache_dir,
        ignore: cli.ignore,
        per_page: cli.per_page,
        ..Config::default()
    };
    anyhow::ensure!(
        !cfg.roots.is_empty(),
        "no roots given: pass --root <dir> or set OASIS_ROOTS"
    );

    match cli.cmd {
        Cmd::Index => {
            let engine = oasis::engine::Engine::build(cfg)?;
            println!("{}", serde_json::to_string_pretty(&engine.stats())?);
        }
        Cmd::Search { query, top_k, json } => {
            let mut engine = oasis::engine::Engine::build(cfg)?;
            let hits = engine.search(&query, top_k)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&hits)?);
            } else {
                for (i, h) in hits.iter().enumerate() {
                    println!(
                        "{:>2}. [{:.4}] {}:{}-{}  {}",
                        i + 1,
                        h.score,
                        h.path,
                        h.line_start,
                        h.line_end,
                        h.heading_path.join(" > ")
                    );
                    println!("    {}", h.snippet.lines().next().unwrap_or(""));
                }
            }
        }
        Cmd::Show { chunk, path } => {
            let engine = oasis::engine::Engine::build(Config {
                lexical_only: true,
                ..cfg
            })?;
            match (chunk, path) {
                (Some(id), _) => {
                    let c = engine
                        .get_chunk(id)
                        .ok_or_else(|| anyhow::anyhow!("no chunk {id}"))?;
                    println!("{}", serde_json::to_string_pretty(c)?);
                }
                (None, Some(p)) => {
                    let doc = engine
                        .get_document(&p)
                        .ok_or_else(|| anyhow::anyhow!("no document {}", p.display()))?;
                    println!("{doc}");
                }
                (None, None) => anyhow::bail!("pass --chunk <id> or --path <file>"),
            }
        }
    }
    Ok(())
}
