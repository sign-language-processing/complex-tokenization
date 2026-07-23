//! Batch corpus ingestion for the default-configuration fast path: one
//! boundary crossing takes the raw text list and does pretokenization
//! (vendored gigatoken O200k — byte-identical to the GPT_PATTERN Split the
//! Python side uses), word dedup, graph building, and Trainer construction
//! entirely in Rust. The Python shim gates this on the default pretokenizer,
//! `utf8_clusters` units, and no registered script handlers; anything else
//! takes the per-document Python path.

use pyo3::prelude::*;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::sync::Arc;
use unicode_segmentation::UnicodeSegmentation;

use crate::graph::GraphV;
use crate::pretok::FastO200kPretokenizer;
use crate::trainer::Trainer;
use crate::units::{extend_cluster_cache, snapshot_cluster_cache, utf8_inner};

/// `utf8_clusters` for one grapheme cluster with no script handlers: a
/// single-char cluster is its UTF-8 graph, a multi-char cluster is a Seq of
/// per-char UTF-8 graphs. Mirrors `resolve_cluster`'s no-handler branch.
fn cluster_graph_pure(cluster: &str) -> GraphV {
    if cluster.chars().nth(1).is_none() {
        return utf8_inner(cluster);
    }
    let char_nodes: Vec<GraphV> = cluster
        .chars()
        .map(|c| utf8_inner(c.encode_utf8(&mut [0u8; 4])))
        .collect();
    GraphV::new_seq(char_nodes)
}

pub(crate) fn build_corpus_graph(texts: &[String], connected: bool) -> GraphV {
    // Pretokenize every document. Match boundaries are char boundaries, so
    // the byte slices are valid &str.
    let doc_tokens: Vec<Vec<&str>> = texts
        .par_iter()
        .map(|t| {
            FastO200kPretokenizer::new(t.as_bytes())
                .map(|p| std::str::from_utf8(p.0).expect("pretoken on char boundary"))
                .collect()
        })
        .collect();

    // Unique words, first-occurrence order.
    let mut word_id: FxHashMap<&str, u32> = FxHashMap::default();
    let mut unique_words: Vec<&str> = Vec::new();
    for tokens in &doc_tokens {
        for &w in tokens {
            word_id.entry(w).or_insert_with(|| {
                unique_words.push(w);
                (unique_words.len() - 1) as u32
            });
        }
    }

    // One graph per unique grapheme cluster, corpus-wide. Existing cluster
    // cache entries are reused and new ones written back, exactly as the
    // per-word `utf8_clusters` path would.
    let mut cluster_of: FxHashMap<&str, GraphV> = FxHashMap::default();
    let cache = snapshot_cluster_cache();
    let mut new_clusters: Vec<(String, GraphV)> = Vec::new();
    for &w in &unique_words {
        for cl in w.graphemes(true) {
            if !cluster_of.contains_key(cl) {
                let g = match cache.get(cl) {
                    Some(g) => g.clone(),
                    None => {
                        let g = cluster_graph_pure(cl);
                        new_clusters.push((cl.to_string(), g.clone()));
                        g
                    }
                };
                cluster_of.insert(cl, g);
            }
        }
    }
    extend_cluster_cache(new_clusters);

    // One graph per unique word, shared by Arc across its occurrences
    // (candidate counting is content-based, so sharing does not affect
    // results; the trainer re-canonicalizes by content anyway).
    let word_graphs: Vec<GraphV> = unique_words
        .par_iter()
        .map(|w| {
            let mut nodes: Vec<GraphV> =
                w.graphemes(true).map(|cl| cluster_of[cl].clone()).collect();
            if nodes.len() == 1 {
                nodes.pop().unwrap()
            } else {
                GraphV::new_seq(nodes)
            }
        })
        .collect();

    // Assemble docs the way `words()` + `Trainer(graphs=...)` would:
    // connected docs are Seqs of their words (a single-word doc is the word
    // itself); disconnected docs flatten into one word-occurrence list.
    let subs: Vec<GraphV> = if connected {
        doc_tokens
            .par_iter()
            .map(|tokens| {
                let mut nodes: Vec<GraphV> = tokens
                    .iter()
                    .map(|w| word_graphs[word_id[w] as usize].clone())
                    .collect();
                if nodes.len() == 1 {
                    nodes.pop().unwrap()
                } else {
                    GraphV::new_seq(nodes)
                }
            })
            .collect()
    } else {
        doc_tokens
            .iter()
            .flat_map(|tokens| tokens.iter())
            .map(|w| word_graphs[word_id[w] as usize].clone())
            .collect()
    };

    GraphV::Unconn(Arc::new(subs))
}

#[pyfunction]
#[pyo3(signature = (texts, connected=false))]
pub fn trainer_from_texts(py: Python<'_>, texts: Vec<String>, connected: bool) -> Trainer {
    let graph = py.allow_threads(|| build_corpus_graph(&texts, connected));
    Trainer::from_graph(graph)
}

#[pyfunction]
pub fn has_cluster_handlers_py() -> bool {
    crate::units::has_cluster_handlers()
}

/// Total pretoken count over `texts` — a benchmark hook for measuring the
/// vendored pretokenizer alone.
#[pyfunction]
pub fn pretokenize_count(py: Python<'_>, texts: Vec<String>) -> usize {
    py.allow_threads(|| {
        texts
            .par_iter()
            .map(|t| FastO200kPretokenizer::new(t.as_bytes()).count())
            .sum()
    })
}
