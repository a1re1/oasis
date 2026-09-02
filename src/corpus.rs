//! Corpus ingestion: walk roots, read markdown, split into heading-aware chunks.
//!
//! Contract:
//! - `load(roots, cfg) -> Vec<Chunk>` walks every root recursively, skipping
//!   hidden dirs, `node_modules`, and `target`.
//! - Chunking is heading-aware: each ATX heading (`#`..`######`) starts a new
//!   section; `heading_path` is the stack of ancestor headings (e.g.
//!   `["Tokio", "Runtime", "Spawning"]`). Sections longer than
//!   `cfg.chunk_tokens` (whitespace tokens, approximate) are split further with
//!   `cfg.chunk_overlap` overlap. Fenced code blocks are never split mid-fence.
//! - `Chunk::id` is dense and stable for the lifetime of an index (`0..n`).

use std::path::{Path, PathBuf};

use globset::{Glob, GlobSet, GlobSetBuilder};
use tracing::warn;

use crate::config::Config;

#[derive(Debug, Clone, serde::Serialize)]
pub struct Chunk {
    pub id: usize,
    /// Path relative to the root it was found under.
    pub path: PathBuf,
    /// Root directory the file was found under.
    pub root: PathBuf,
    /// Ancestor heading titles, outermost first.
    pub heading_path: Vec<String>,
    /// Chunk body (markdown source, trimmed).
    pub text: String,
    /// 1-based line range in the source file.
    pub line_start: usize,
    pub line_end: usize,
}

impl Chunk {
    /// Text handed to the embedder / tokenizer: headings + body, so that a
    /// chunk that only says "see above" still carries its context.
    pub fn indexable_text(&self) -> String {
        if self.heading_path.is_empty() {
            return self.text.clone();
        }
        format!("{}\n{}", self.heading_path.join(" > "), self.text)
    }
}

/// Walk `roots` and return all chunks. Files that fail to read are logged and skipped.
pub fn load(roots: &[PathBuf], cfg: &Config) -> anyhow::Result<Vec<Chunk>> {
    let ignore = compile_ignore(&cfg.ignore);
    let mut files: Vec<(PathBuf, PathBuf)> = Vec::new(); // (root, file path)
    for root in roots {
        let root = root.clone();
        for entry in walkdir::WalkDir::new(&root).into_iter().filter_entry(|e| {
            // The root itself is always kept (temp dirs are often hidden).
            e.depth() == 0 || !is_skipped_name(e.file_name())
        }) {
            let entry = match entry {
                Ok(e) => e,
                Err(err) => {
                    warn!("skipping unreadable path under {}: {err}", root.display());
                    continue;
                }
            };
            // Ignore globs are matched against the path relative to the root;
            // directories are checked too so `log/**`-style rules prune whole trees.
            if entry.depth() > 0 && ignore.is_match(entry.path().strip_prefix(&root).unwrap_or(entry.path())) {
                continue;
            }
            if !entry.file_type().is_file() {
                continue;
            }
            let ext = entry.path().extension().and_then(|e| e.to_str());
            if !matches!(ext, Some("md") | Some("mdx") | Some("markdown")) {
                continue;
            }
            files.push((root.clone(), entry.path().to_path_buf()));
        }
    }
    // Deterministic chunk ids: sort by root, then file path.
    files.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    let mut chunks = Vec::new();
    for (root, path) in files {
        let rel = path.strip_prefix(&root).unwrap_or(&path).to_path_buf();
        let source = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(err) => {
                warn!("skipping unreadable file {}: {err}", path.display());
                continue;
            }
        };
        for mut chunk in chunk_markdown(&root, &rel, &source, cfg) {
            chunk.id = chunks.len();
            chunks.push(chunk);
        }
    }
    Ok(chunks)
}

/// Directory / file names we never descend into or index.
fn is_skipped_name(name: &std::ffi::OsStr) -> bool {
    let name = name.to_string_lossy();
    name == "node_modules" || name == "target" || name.starts_with('.')
}

/// Compile `cfg.ignore` globs (matched against paths relative to the root).
/// Invalid patterns are logged and skipped rather than failing the whole walk.
fn compile_ignore(patterns: &[String]) -> GlobSet {
    let mut builder = GlobSetBuilder::new();
    for pat in patterns {
        match Glob::new(pat) {
            Ok(glob) => {
                builder.add(glob);
            }
            Err(err) => warn!("skipping invalid --ignore glob {pat:?}: {err}"),
        }
    }
    builder.build().unwrap_or_else(|_| GlobSet::empty())
}

struct RawSection {
    heading_path: Vec<String>,
    /// 1-based line number of the first line of the section.
    start_line: usize,
    /// Raw source lines, heading line included.
    lines: Vec<String>,
}

/// Chunk a single markdown document. Exposed for tests.
pub fn chunk_markdown(root: &Path, rel_path: &Path, source: &str, cfg: &Config) -> Vec<Chunk> {
    let mut out = Vec::new();
    for section in split_sections(source) {
        if section.lines.iter().all(|l| l.trim().is_empty()) {
            continue;
        }
        for (offset, piece) in split_oversize(&section.lines, cfg.chunk_tokens, cfg.chunk_overlap) {
            // Trim blank edges, keeping the 1-based line numbers honest.
            let lead = piece.iter().take_while(|l| l.trim().is_empty()).count();
            let trail = piece
                .iter()
                .rev()
                .take_while(|l| l.trim().is_empty())
                .count();
            if lead + trail >= piece.len() {
                continue;
            }
            let body = &piece[lead..piece.len() - trail];
            let text = body.join("\n");
            let first_line = section.start_line + offset + lead;
            out.push(Chunk {
                id: out.len(), // reassigned globally by `load`
                path: rel_path.to_path_buf(),
                root: root.to_path_buf(),
                heading_path: section.heading_path.clone(),
                text,
                line_start: first_line,
                line_end: first_line + body.len() - 1,
            });
        }
    }
    out
}

/// Split a document into raw sections at ATX headings (outside fenced code).
/// Content before the first heading forms a section with an empty heading path.
fn split_sections(source: &str) -> Vec<RawSection> {
    let mut sections: Vec<RawSection> = Vec::new();
    let mut stack: Vec<(usize, String)> = Vec::new(); // (level, title)
    let mut current = RawSection {
        heading_path: Vec::new(),
        start_line: 1,
        lines: Vec::new(),
    };
    let mut fence: Option<(char, usize)> = None; // (fence char, run length)

    for (idx, line) in source.lines().enumerate() {
        let trimmed = line.trim_start();
        if let Some((ch, len)) = fence {
            if is_fence_close(trimmed, ch, len) {
                fence = None;
            }
            current.lines.push(line.to_string());
            continue;
        }
        if let Some(open) = fence_open(trimmed) {
            fence = Some(open);
            current.lines.push(line.to_string());
            continue;
        }
        if let Some((level, title)) = parse_atx_heading(line) {
            sections.push(std::mem::replace(
                &mut current,
                RawSection {
                    heading_path: Vec::new(),
                    start_line: idx + 1,
                    lines: vec![line.to_string()],
                },
            ));
            while stack.last().is_some_and(|&(l, _)| l >= level) {
                stack.pop();
            }
            stack.push((level, title.clone()));
            current.heading_path = stack.iter().map(|(_, t)| t.clone()).collect();
            continue;
        }
        current.lines.push(line.to_string());
    }
    sections.push(current);
    sections
}

/// Split a section's lines into pieces of at most `max_tokens` whitespace
/// tokens (a single over-long line is kept whole) with roughly `overlap`
/// tokens of overlap, never cutting inside a fenced code block.
fn split_oversize(
    lines: &[String],
    max_tokens: usize,
    overlap: usize,
) -> Vec<(usize, Vec<String>)> {
    let n = lines.len();
    let toks: Vec<usize> = lines.iter().map(|l| l.split_whitespace().count()).collect();
    let total: usize = toks.iter().sum();
    if max_tokens == 0 || total <= max_tokens {
        return vec![(0, lines.to_vec())];
    }
    // safe[i] == true iff a cut after `i` lines does not split a fence.
    let mut safe = vec![true; n + 1];
    let mut fence: Option<(char, usize)> = None;
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        match fence {
            Some((ch, len)) => {
                if is_fence_close(trimmed, ch, len) {
                    fence = None;
                }
            }
            None => fence = fence_open(trimmed),
        }
        safe[i + 1] = fence.is_none();
    }
    // Smallest `e > start` such that lines[start..e] holds <= max_tokens tokens
    // (always at least one line), pushed forward to the next fence-safe cut.
    let budget_end = |start: usize| -> usize {
        let mut e = start;
        let mut used = 0;
        while e < n && (e == start || used + toks[e] <= max_tokens) {
            used += toks[e];
            e += 1;
        }
        while e < n && !safe[e] {
            e += 1;
        }
        e.min(n)
    };
    // Walk back from `end` until at least `overlap` tokens are covered.
    let overlap_start = |start: usize, end: usize| -> usize {
        let mut s = end;
        let mut covered = 0;
        while s > start + 1 && covered < overlap {
            covered += toks[s - 1];
            s -= 1;
        }
        s
    };

    let mut pieces: Vec<(usize, Vec<String>)> = Vec::new();
    let mut start = 0usize;
    loop {
        let end = budget_end(start);
        pieces.push((start, lines[start..end].to_vec()));
        if end >= n {
            break;
        }
        let s = if overlap == 0 {
            end
        } else {
            overlap_start(start, end)
        };
        // guarantee progress even with overlap >= chunk_tokens
        start = if s <= start { end } else { s };
    }
    pieces
}

/// An opening fence: a run of 3+ backticks or tildes (info string allowed).
fn fence_open(trimmed: &str) -> Option<(char, usize)> {
    for ch in ['`', '~'] {
        let (run, _) = fence_run(trimmed, ch);
        if run >= 3 {
            return Some((ch, run));
        }
    }
    None
}

/// A closing fence: a run of the same char, at least as long as the opener,
/// with nothing else on the line.
fn is_fence_close(trimmed: &str, ch: char, len: usize) -> bool {
    let (run, byte) = fence_run(trimmed, ch);
    run >= len && trimmed[byte..].trim().is_empty()
}

/// (run length, byte offset just past the run) of `ch` at the start of `s`.
fn fence_run(s: &str, ch: char) -> (usize, usize) {
    let (mut run, mut byte) = (0usize, 0usize);
    for (i, c) in s.char_indices() {
        if c != ch {
            break;
        }
        run += 1;
        byte = i + c.len_utf8();
    }
    (run, byte)
}

/// Parse an ATX heading line: 1-6 `#`s, then whitespace, then the title.
fn parse_atx_heading(line: &str) -> Option<(usize, String)> {
    let trimmed = line.trim_start();
    let level = trimmed.chars().take_while(|&c| c == '#').count();
    if level == 0 || level > 6 {
        return None;
    }
    let rest = &trimmed[level..];
    if !rest.is_empty() && !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let title = rest.trim().trim_end_matches('#').trim().to_string();
    Some((level, title))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heading_path_nests_and_line_numbers_are_1_based() {
        let src = "# Tokio\nintro\n## Runtime\nruntime body\n### Spawning\nspawn body\n## Shutdown\nbye\n";
        let chunks = chunk_markdown(Path::new("/r"), Path::new("t.md"), src, &Config::default());
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0].heading_path, vec!["Tokio"]);
        assert_eq!(chunks[1].heading_path, vec!["Tokio", "Runtime"]);
        assert_eq!(chunks[2].heading_path, vec!["Tokio", "Runtime", "Spawning"]);
        assert_eq!(chunks[3].heading_path, vec!["Tokio", "Shutdown"]);
        assert_eq!(chunks[2].line_start, 5);
        assert_eq!(chunks[2].line_end, 6);
        assert_eq!(chunks[3].line_start, 7);
        assert_eq!(chunks[3].line_end, 8);
        assert_eq!(chunks[0].text, "# Tokio\nintro");
        assert_eq!(chunks[2].text, "### Spawning\nspawn body");
    }

    #[test]
    fn preamble_has_empty_heading_path() {
        let src = "preamble text\n# H\nbody\n";
        let chunks = chunk_markdown(Path::new("/r"), Path::new("t.md"), src, &Config::default());
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].heading_path.is_empty());
        assert_eq!(chunks[0].text, "preamble text");
        assert_eq!(chunks[0].line_start, 1);
    }

    #[test]
    fn indexable_text_prefixes_heading_path() {
        let src = "# A\n## B\nbody\n";
        let chunks = chunk_markdown(Path::new("/r"), Path::new("t.md"), src, &Config::default());
        assert_eq!(chunks[0].indexable_text(), "A\n# A");
        assert_eq!(chunks.last().unwrap().indexable_text(), "A > B\n## B\nbody");
    }

    #[test]
    fn code_fence_is_never_split() {
        let cfg = Config {
            chunk_tokens: 4,
            chunk_overlap: 0,
            ..Config::default()
        };
        let mut src = String::from("# Code\n```rust\n");
        for i in 0..10 {
            src.push_str(&format!("let x{i} = {i};\n"));
        }
        src.push_str("```\nafter the fence\n");
        let chunks = chunk_markdown(Path::new("/r"), Path::new("t.md"), &src, &cfg);
        // Every chunk containing the opening fence must also contain the close.
        for c in &chunks {
            if c.text.contains("```rust") {
                assert!(c.text.contains("let x9"), "fence split: {}", c.text);
            }
        }
        assert!(chunks.iter().any(|c| c.text.contains("after the fence")));
    }

    #[test]
    fn tilde_fence_is_never_split() {
        let cfg = Config {
            chunk_tokens: 3,
            chunk_overlap: 0,
            ..Config::default()
        };
        let src =
            "# T\n~~~\nlong code line one\nlong code line two\nlong code line three\n~~~\ntail\n";
        let chunks = chunk_markdown(Path::new("/r"), Path::new("t.md"), src, &cfg);
        assert!(chunks.iter().any(|c| {
            c.text.contains("~~~") && c.text.contains("one") && c.text.contains("three")
        }));
    }

    #[test]
    fn oversize_section_splits_with_overlap() {
        let cfg = Config {
            chunk_tokens: 5,
            chunk_overlap: 2,
            ..Config::default()
        };
        let mut src = String::from("# Top\n");
        for i in 0..20 {
            src.push_str(&format!("line{i:02}\n"));
        }
        let chunks = chunk_markdown(Path::new("/r"), Path::new("t.md"), &src, &cfg);
        assert!(
            chunks.len() >= 3,
            "expected several chunks, got {}",
            chunks.len()
        );
        assert_eq!(chunks[0].line_start, 1); // heading line
        for w in chunks.windows(2) {
            assert_eq!(
                w[1].line_start,
                w[0].line_end + 1 - 2,
                "overlap mismatch: {:?} vs {:?}",
                (w[0].line_start, w[0].line_end),
                (w[1].line_start, w[1].line_end)
            );
        }
        assert_eq!(chunks.last().unwrap().line_end, 21);
    }

    #[test]
    fn budget_is_whitespace_tokens_not_lines() {
        let cfg = Config {
            chunk_tokens: 6,
            chunk_overlap: 0,
            ..Config::default()
        };
        // 4 lines x 4 tokens = 16 tokens; a line budget of 6 would keep one chunk.
        let src = "# H\nw w w w\nw w w w\nw w w w\nw w w w\n";
        let chunks = chunk_markdown(Path::new("/r"), Path::new("t.md"), src, &cfg);
        assert!(chunks.len() >= 3, "got {}", chunks.len());
        for c in &chunks {
            let toks = c.text.split_whitespace().count();
            assert!(
                toks <= 6 || c.text.lines().count() == 1,
                "{toks} tokens: {}",
                c.text
            );
        }
    }

    #[test]
    fn load_walks_skips_and_sorts() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        std::fs::create_dir_all(root.join("sub")).unwrap();
        std::fs::create_dir_all(root.join(".hidden")).unwrap();
        std::fs::create_dir_all(root.join("node_modules")).unwrap();
        std::fs::create_dir_all(root.join("target")).unwrap();
        std::fs::write(root.join("b.md"), "# B\nb text\n").unwrap();
        std::fs::write(root.join("sub/a.markdown"), "# A\na text\n").unwrap();
        std::fs::write(root.join("sub/n.txt"), "not markdown\n").unwrap();
        std::fs::write(root.join(".hidden/h.md"), "# H\n").unwrap();
        std::fs::write(root.join("node_modules/n.md"), "# N\n").unwrap();
        std::fs::write(root.join("target/t.md"), "# T\n").unwrap();

        let chunks = load(&[root.to_path_buf()], &Config::default()).unwrap();
        let paths: Vec<String> = chunks
            .iter()
            .map(|c| c.path.to_string_lossy().into())
            .collect();
        assert_eq!(paths, vec!["b.md", "sub/a.markdown"]);
        assert_eq!(chunks[0].id, 0);
        assert_eq!(chunks[1].id, 1);
        assert_eq!(chunks[0].root, root);
    }

    #[test]
    fn load_skips_ignored_globs() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        std::fs::create_dir_all(root.join("a")).unwrap();
        std::fs::write(root.join("index.md"), "# Index\nindex text\n").unwrap();
        std::fs::write(root.join("a/b.md"), "# B\nb text\n").unwrap();

        let cfg = Config {
            ignore: vec!["index.md".to_string()],
            ..Config::default()
        };
        let chunks = load(&[root.to_path_buf()], &cfg).unwrap();
        let paths: Vec<String> = chunks
            .iter()
            .map(|c| c.path.to_string_lossy().into())
            .collect();
        assert_eq!(paths, vec!["a/b.md"]);

        // No ignore patterns -> both files are chunked as before.
        let chunks = load(&[root.to_path_buf()], &Config::default()).unwrap();
        assert_eq!(chunks.len(), 2);
    }

    #[test]
    fn heading_needs_whitespace_after_hashes() {
        let src = "#notahead\n# Real\nbody\n";
        let chunks = chunk_markdown(Path::new("/r"), Path::new("t.md"), src, &Config::default());
        // `#notahead` is not a heading -> it stays in the preamble section.
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].heading_path.is_empty());
        assert!(chunks[0].text.contains("#notahead"));
        assert_eq!(chunks[1].heading_path, vec!["Real"]);
    }
}
