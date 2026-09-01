use std::path::Path;

use oasis::{config::Config, engine::Engine};

const FILES: &[(&str, &str)] = &[
    (
        "rust/tokio.md",
        "# Tokio\n\n## Runtime\n\nThe tokio runtime drives async tasks.\n\n## Spawning\n\nUse `tokio::spawn` to spawn a task onto the runtime. A spawned task runs concurrently.\n",
    ),
    (
        "rust/errors.md",
        "# Error handling\n\nUse `anyhow` for applications and `thiserror` for libraries. The `?` operator propagates errors.\n",
    ),
    (
        "git/rebase.md",
        "# Git rebase\n\n## Interactive\n\n`git rebase -i` opens an interactive rebase to squash, reorder, or edit commits.\n",
    ),
    (
        "db/sqlite.md",
        "# SQLite indexes\n\nCreate an index with `CREATE INDEX`. Covering indexes avoid table lookups.\n",
    ),
    (
        "misc/notes.md",
        "# Notes\n\nRandom notes about shell aliases and dotfiles.\n",
    ),
];

fn fixture() -> (tempfile::TempDir, Engine) {
    let dir = tempfile::tempdir().unwrap();
    for (rel, body) in FILES {
        let p = dir.path().join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, body).unwrap();
    }
    let cfg = Config {
        roots: vec![dir.path().to_path_buf()],
        lexical_only: true,
        ..Config::default()
    };
    let engine = Engine::build(cfg).unwrap();
    (dir, engine)
}

#[test]
fn tokio_query_hits_tokio_file_first() {
    let (_d, mut e) = fixture();
    let hits = e.search("tokio spawn task", 3).unwrap();
    assert!(!hits.is_empty());
    assert_eq!(hits[0].path, "rust/tokio.md", "{hits:?}");
    assert!(
        hits[0].heading_path.contains(&"Spawning".to_string()),
        "{hits:?}"
    );
}

#[test]
fn git_query_hits_git_file_first() {
    let (_d, mut e) = fixture();
    let hits = e.search("git rebase interactive", 3).unwrap();
    assert_eq!(hits[0].path, "git/rebase.md", "{hits:?}");
}

#[test]
fn get_document_reconstructs_file() {
    let (_d, e) = fixture();
    let doc = e.get_document(Path::new("rust/tokio.md")).unwrap();
    assert!(doc.contains("tokio runtime"));
    assert!(doc.contains("tokio::spawn"));
    assert!(e.get_document(Path::new("nope.md")).is_none());
    let stats = e.stats();
    assert_eq!(stats["files"], 5);
    assert_eq!(stats["dense"], false);
}

#[test]
fn empty_query_returns_nothing() {
    let (_d, mut e) = fixture();
    assert!(e.search("   ", 5).unwrap().is_empty());
}
