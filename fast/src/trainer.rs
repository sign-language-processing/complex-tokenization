use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::sync::Arc;

use crate::graph::{graphv_to_pyobject, pyobject_to_graphv, GraphV};
use crate::units::{apply_merge_to_cluster_cache, replace_word_cache, snapshot_word_cache};
use std::collections::HashMap;

/// Pick the merge with the highest score: (len - 1) * count.
/// Ties broken by bytes (deterministic, order-independent).
fn pick_best(map: &FxHashMap<Vec<GraphV>, usize>) -> Option<(Vec<GraphV>, usize)> {
    let mut best: Option<(&Vec<GraphV>, usize, Vec<Vec<u8>>)> = None;
    for (key, &count) in map {
        let score = (key.len() - 1) * count;
        if score == 0 {
            continue;
        }
        let dominated = match &best {
            Some((_, bs, _)) => score < *bs,
            None => false,
        };
        if dominated {
            continue;
        }
        let bytes_key: Vec<Vec<u8>> = key.iter().map(|g| g.to_bytes()).collect();
        let replace = match &best {
            None => true,
            Some((_, bs, bb)) => score > *bs || (score == *bs && bytes_key > *bb),
        };
        if replace {
            best = Some((key, score, bytes_key));
        }
    }
    best.map(|(k, _, _)| (k.clone(), map[k]))
}

fn make_token(nodes: &[GraphV]) -> GraphV {
    nodes
        .iter()
        .skip(1)
        .fold(nodes[0].clone(), |acc, n| acc.add(n))
}

fn count_merges_parallel(subs: &[GraphV], active: &[bool]) -> FxHashMap<Vec<GraphV>, usize> {
    let n_threads = rayon::current_num_threads().max(1);
    let chunk_size = (subs.len() / n_threads).max(1);

    let chunk_maps: Vec<FxHashMap<Vec<GraphV>, usize>> = subs
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            let mut local: FxHashMap<Vec<GraphV>, usize> = FxHashMap::default();
            let base = chunk_idx * chunk_size;
            for (offset, sg) in chunk.iter().enumerate() {
                if active[base + offset] {
                    sg.emit_merges(&mut |m| *local.entry(m).or_insert(0) += 1);
                }
            }
            local
        })
        .collect();

    let total: usize = chunk_maps.iter().map(|m| m.len()).sum();
    let mut global: FxHashMap<Vec<GraphV>, usize> =
        FxHashMap::with_capacity_and_hasher(total, Default::default());
    for partial in chunk_maps {
        for (merge, count) in partial {
            *global.entry(merge).or_insert(0) += count;
        }
    }
    global
}

fn apply_merge_parallel(
    subs: &mut [GraphV],
    active: &mut [bool],
    token: &GraphV,
    merge: &[GraphV],
) {
    let changes: Vec<(usize, GraphV)> = subs
        .par_iter()
        .enumerate()
        .filter_map(|(i, sg)| {
            if !active[i] {
                return None;
            }
            sg.try_merge(token, merge).map(|new_sg| (i, new_sg))
        })
        .collect();

    for (i, new_sg) in changes {
        if matches!(&new_sg, GraphV::Node(_)) {
            active[i] = false;
        }
        subs[i] = new_sg;
    }
}

fn flatten_unconn(g: GraphV) -> GraphV {
    if let GraphV::Unconn(subs) = &g {
        let mut flat = Vec::new();
        for sub in subs.as_ref() {
            if let GraphV::Unconn(inner) = sub {
                flat.extend(inner.as_ref().iter().cloned());
            } else {
                flat.push(sub.clone());
            }
        }
        GraphV::Unconn(Arc::new(flat))
    } else {
        g
    }
}

fn do_merge_step(
    subs: &mut [GraphV],
    active: &mut [bool],
    verbose: bool,
) -> Option<(GraphV, Vec<GraphV>)> {
    let counts = count_merges_parallel(subs, active);
    let (nodes, count) = pick_best(&counts)?;
    let token = make_token(&nodes);
    if verbose {
        println!("Merging {:?} count={}", nodes, count);
    }
    apply_merge_parallel(subs, active, &token, &nodes);
    apply_merge_to_cluster_cache(&token, &nodes);
    Some((token, nodes))
}

fn train_unconn(
    subs: &mut Vec<GraphV>,
    range_start: usize,
    num_merges: usize,
    verbose: bool,
) -> Vec<(GraphV, Vec<GraphV>)> {
    let mut active: Vec<bool> = subs
        .iter()
        .map(|sg| !matches!(sg, GraphV::Node(_)))
        .collect();
    let mut pending = Vec::new();

    for _ in range_start..num_merges {
        match do_merge_step(subs, &mut active, verbose) {
            Some(merge) => pending.push(merge),
            None => break,
        }
    }
    pending
}

fn train_single(
    graph: &mut GraphV,
    range_start: usize,
    num_merges: usize,
    verbose: bool,
) -> Vec<(GraphV, Vec<GraphV>)> {
    let mut pending = Vec::new();

    for _ in range_start..num_merges {
        let mut counts: FxHashMap<Vec<GraphV>, usize> = FxHashMap::default();
        graph.emit_merges(&mut |m| *counts.entry(m).or_insert(0) += 1);
        let Some((nodes, count)) = pick_best(&counts) else {
            break;
        };
        let token = make_token(&nodes);
        if verbose {
            println!("Merging {:?} count={}", nodes, count);
        }
        *graph = graph.merge(&token, &nodes);
        apply_merge_to_cluster_cache(&token, &nodes);
        pending.push((token, nodes));
    }
    pending
}

// ---- Streaming trainer with word-level candidate caching ----

struct WordEntry {
    graph: GraphV,
    candidates: FxHashMap<Vec<GraphV>, usize>,
    freq: usize,
}

impl WordEntry {
    fn new(graph: GraphV, freq: usize) -> Self {
        let mut candidates = FxHashMap::default();
        graph.emit_merges(&mut |m| *candidates.entry(m).or_insert(0) += 1);
        WordEntry { graph, candidates, freq }
    }

    fn apply_merge(&mut self, token: &GraphV, merge: &[GraphV]) {
        self.graph = self.graph.merge(token, merge);
        self.candidates.clear();
        self.graph.emit_merges(&mut |m| *self.candidates.entry(m).or_insert(0) += 1);
    }
}

fn build_word_entries(
    doc_words: &[Vec<String>],
    cache: &HashMap<String, GraphV>,
) -> Vec<WordEntry> {
    let mut freq_map: FxHashMap<String, usize> = FxHashMap::default();
    for words in doc_words {
        for w in words {
            *freq_map.entry(w.clone()).or_insert(0) += 1;
        }
    }
    freq_map
        .into_iter()
        .filter_map(|(w, freq)| {
            cache.get(w.as_str()).map(|g| WordEntry::new(g.clone(), freq))
        })
        .collect()
}

fn build_doc_graph(words: &[String], cache: &HashMap<String, GraphV>, connected: bool) -> GraphV {
    let word_graphs: Vec<GraphV> = words
        .iter()
        .filter_map(|w| cache.get(w.as_str()).cloned())
        .collect();
    if word_graphs.len() == 1 {
        return word_graphs.into_iter().next().unwrap();
    }
    if connected {
        GraphV::new_seq(word_graphs)
    } else {
        GraphV::Unconn(Arc::new(word_graphs))
    }
}

// Disconnected: use word-level candidate caching (no graph assembly)
fn train_streaming_disconnected(
    entries: &mut Vec<WordEntry>,
    range_start: usize,
    num_merges: usize,
    verbose: bool,
) -> Vec<(GraphV, Vec<GraphV>)> {
    let mut pending = Vec::new();

    for _ in range_start..num_merges {
        // Aggregate cached candidates × freq (parallel)
        let n_threads = rayon::current_num_threads().max(1);
        let chunk_size = (entries.len() / n_threads).max(1);

        let chunk_maps: Vec<FxHashMap<Vec<GraphV>, usize>> = entries
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut local: FxHashMap<Vec<GraphV>, usize> = FxHashMap::default();
                for entry in chunk {
                    for (merge, &count) in &entry.candidates {
                        *local.entry(merge.clone()).or_insert(0) += count * entry.freq;
                    }
                }
                local
            })
            .collect();

        let total: usize = chunk_maps.iter().map(|m| m.len()).sum();
        let mut global: FxHashMap<Vec<GraphV>, usize> =
            FxHashMap::with_capacity_and_hasher(total, Default::default());
        for partial in chunk_maps {
            for (merge, count) in partial {
                *global.entry(merge).or_insert(0) += count;
            }
        }

        let Some((nodes, count)) = pick_best(&global) else {
            break;
        };
        let token = make_token(&nodes);
        if verbose {
            println!("Merging {:?} count={}", nodes, count);
        }

        // Update only affected entries (parallel)
        entries.par_iter_mut().for_each(|entry| {
            if entry.candidates.contains_key(&nodes) {
                entry.apply_merge(&token, &nodes);
            }
        });

        pending.push((token, nodes));
    }
    pending
}

// Connected: must assemble doc graphs for cross-word pairs
fn train_streaming_connected(
    doc_words: &[Vec<String>],
    cache: &mut HashMap<String, GraphV>,
    range_start: usize,
    num_merges: usize,
    verbose: bool,
) -> Vec<(GraphV, Vec<GraphV>)> {
    let mut pending = Vec::new();

    for _ in range_start..num_merges {
        let n_threads = rayon::current_num_threads().max(1);
        let chunk_size = (doc_words.len() / n_threads).max(1);

        let chunk_maps: Vec<FxHashMap<Vec<GraphV>, usize>> = doc_words
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut local: FxHashMap<Vec<GraphV>, usize> = FxHashMap::default();
                for words in chunk {
                    let doc = build_doc_graph(words, cache, true);
                    doc.emit_merges(&mut |m| *local.entry(m).or_insert(0) += 1);
                }
                local
            })
            .collect();

        let total: usize = chunk_maps.iter().map(|m| m.len()).sum();
        let mut global: FxHashMap<Vec<GraphV>, usize> =
            FxHashMap::with_capacity_and_hasher(total, Default::default());
        for partial in chunk_maps {
            for (merge, count) in partial {
                *global.entry(merge).or_insert(0) += count;
            }
        }

        let Some((nodes, count)) = pick_best(&global) else {
            break;
        };
        let token = make_token(&nodes);
        if verbose {
            println!("Merging {:?} count={}", nodes, count);
        }

        for graph in cache.values_mut() {
            if let Some(new_g) = graph.try_merge(&token, &nodes) {
                *graph = new_g;
            }
        }

        pending.push((token, nodes));
    }
    pending
}

fn train_streaming(
    doc_words: &[Vec<String>],
    connected: bool,
    range_start: usize,
    num_merges: usize,
    verbose: bool,
) -> Vec<(GraphV, Vec<GraphV>)> {
    let mut cache = snapshot_word_cache();

    let pending = if connected {
        train_streaming_connected(doc_words, &mut cache, range_start, num_merges, verbose)
    } else {
        let mut entries = build_word_entries(doc_words, &cache);
        let result = train_streaming_disconnected(&mut entries, range_start, num_merges, verbose);
        // Write back updated graphs to cache
        cache.clear();
        // We lost the string keys during entry building, so just update the word cache
        // from the global word cache + apply all pending merges
        let mut wc = snapshot_word_cache();
        for (token, nodes) in &result {
            for graph in wc.values_mut() {
                if let Some(new_g) = graph.try_merge(token, nodes) {
                    *graph = new_g;
                }
            }
        }
        replace_word_cache(wc);
        return result;
    };

    replace_word_cache(cache);
    pending
}

fn train_streaming_with_counts(
    doc_words: &[Vec<String>],
    connected: bool,
    range_start: usize,
    num_merges: usize,
    sample_every: usize,
) -> (Vec<(GraphV, Vec<GraphV>)>, Vec<usize>, Vec<usize>) {
    let mut cache = snapshot_word_cache();

    if !connected {
        let mut entries = build_word_entries(doc_words, &cache);
        let word_freqs: Vec<usize> = entries.iter().map(|e| e.freq).collect();

        let node_count = |entries: &[WordEntry]| -> usize {
            entries.par_iter().map(|e| e.graph.node_count() * e.freq).sum()
        };

        let mut pending = Vec::new();
        let mut xs = vec![0usize];
        let mut ys = vec![node_count(&entries)];

        for i in range_start..num_merges {
            let n_threads = rayon::current_num_threads().max(1);
            let chunk_size = (entries.len() / n_threads).max(1);

            let chunk_maps: Vec<FxHashMap<Vec<GraphV>, usize>> = entries
                .par_chunks(chunk_size)
                .map(|chunk| {
                    let mut local: FxHashMap<Vec<GraphV>, usize> = FxHashMap::default();
                    for entry in chunk {
                        for (merge, &count) in &entry.candidates {
                            *local.entry(merge.clone()).or_insert(0) += count * entry.freq;
                        }
                    }
                    local
                })
                .collect();

            let total: usize = chunk_maps.iter().map(|m| m.len()).sum();
            let mut global: FxHashMap<Vec<GraphV>, usize> =
                FxHashMap::with_capacity_and_hasher(total, Default::default());
            for partial in chunk_maps {
                for (merge, count) in partial {
                    *global.entry(merge).or_insert(0) += count;
                }
            }

            let Some((nodes, _)) = pick_best(&global) else { break };
            let token = make_token(&nodes);

            entries.par_iter_mut().for_each(|entry| {
                if entry.candidates.contains_key(&nodes) {
                    entry.apply_merge(&token, &nodes);
                }
            });

            pending.push((token, nodes));

            let merge_num = i + 1;
            if merge_num % sample_every == 0 || merge_num == num_merges {
                xs.push(merge_num);
                ys.push(node_count(&entries));
            }
        }

        // Update word cache
        let mut wc = snapshot_word_cache();
        for (token, nodes) in &pending {
            for graph in wc.values_mut() {
                if let Some(new_g) = graph.try_merge(token, nodes) {
                    *graph = new_g;
                }
            }
        }
        replace_word_cache(wc);

        return (pending, xs, ys);
    }

    // Connected: fall back to doc graph assembly
    let node_count_connected = |cache: &HashMap<String, GraphV>| -> usize {
        doc_words.par_iter()
            .map(|words| build_doc_graph(words, cache, true).node_count())
            .sum()
    };

    let mut pending = Vec::new();
    let mut xs = vec![0usize];
    let mut ys = vec![node_count_connected(&cache)];

    for i in range_start..num_merges {
        let n_threads = rayon::current_num_threads().max(1);
        let chunk_size = (doc_words.len() / n_threads).max(1);

        let chunk_maps: Vec<FxHashMap<Vec<GraphV>, usize>> = doc_words
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut local: FxHashMap<Vec<GraphV>, usize> = FxHashMap::default();
                for words in chunk {
                    let doc = build_doc_graph(words, &cache, true);
                    doc.emit_merges(&mut |m| *local.entry(m).or_insert(0) += 1);
                }
                local
            })
            .collect();

        let total: usize = chunk_maps.iter().map(|m| m.len()).sum();
        let mut global: FxHashMap<Vec<GraphV>, usize> =
            FxHashMap::with_capacity_and_hasher(total, Default::default());
        for partial in chunk_maps {
            for (merge, count) in partial {
                *global.entry(merge).or_insert(0) += count;
            }
        }

        let Some((nodes, _)) = pick_best(&global) else { break };
        let token = make_token(&nodes);

        for graph in cache.values_mut() {
            if let Some(new_g) = graph.try_merge(&token, &nodes) {
                *graph = new_g;
            }
        }

        pending.push((token, nodes));

        let merge_num = i + 1;
        if merge_num % sample_every == 0 || merge_num == num_merges {
            xs.push(merge_num);
            ys.push(node_count_connected(&cache));
        }
    }

    replace_word_cache(cache);
    (pending, xs, ys)
}

#[pyclass]
pub struct Trainer {
    pub graph: GraphV,
    pub merges: Vec<(GraphV, Vec<GraphV>)>,
    doc_words: Option<Vec<Vec<String>>>,
    streaming_connected: bool,
}

#[pymethods]
impl Trainer {
    #[new]
    #[pyo3(signature = (graph=None, graphs=None))]
    fn new(
        graph: Option<&Bound<'_, PyAny>>,
        graphs: Option<&Bound<'_, PyTuple>>,
    ) -> PyResult<Self> {
        match (graph, graphs) {
            (None, None) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Must provide either graph or graphs",
            )),
            (Some(_), Some(_)) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Must provide either graph or graphs, not both",
            )),
            (Some(g), None) => Ok(Trainer {
                graph: pyobject_to_graphv(g)?,
                merges: Vec::new(),
                doc_words: None,
                streaming_connected: false,
            }),
            (None, Some(gs)) => {
                let subs: Vec<GraphV> = gs
                    .iter()
                    .map(|item| pyobject_to_graphv(&item))
                    .collect::<PyResult<_>>()?;
                Ok(Trainer {
                    graph: flatten_unconn(GraphV::Unconn(Arc::new(subs))),
                    merges: Vec::new(),
                    doc_words: None,
                    streaming_connected: false,
                })
            }
        }
    }

    /// Set pretokenized word lists for streaming mode.
    /// Each element is a list of word strings for one document.
    /// The cluster cache must be warmed (call utf8_clusters for each word first).
    #[pyo3(signature = (doc_words, connected=false))]
    fn set_streaming(
        &mut self,
        doc_words: Vec<Vec<String>>,
        connected: bool,
    ) {
        self.doc_words = Some(doc_words);
        self.streaming_connected = connected;
    }

    #[pyo3(signature = (num_merges=100, draw=false, verbose=false, progress=false))]
    fn train(
        &mut self,
        py: Python<'_>,
        num_merges: usize,
        draw: bool,
        verbose: bool,
        progress: bool,
    ) -> PyResult<()> {
        let _ = (draw, progress);
        let range_start = self.merges.len();
        if range_start >= num_merges {
            return Ok(());
        }

        if let Some(ref doc_words) = self.doc_words {
            let connected = self.streaming_connected;
            let pending =
                py.allow_threads(|| train_streaming(doc_words, connected, range_start, num_merges, verbose));
            self.merges.extend(pending);
            return Ok(());
        }

        let graph = &mut self.graph;
        let pending = py.allow_threads(|| {
            if let GraphV::Unconn(subs_arc) = graph {
                let subs = Arc::make_mut(subs_arc);
                train_unconn(subs, range_start, num_merges, verbose)
            } else {
                train_single(graph, range_start, num_merges, verbose)
            }
        });

        self.merges.extend(pending);
        Ok(())
    }

    #[pyo3(signature = (num_merges, sample_every=1))]
    fn train_with_counts(
        &mut self,
        py: Python<'_>,
        num_merges: usize,
        sample_every: usize,
    ) -> PyResult<(Vec<usize>, Vec<usize>)> {
        let range_start = self.merges.len();

        if let Some(ref doc_words) = self.doc_words {
            let connected = self.streaming_connected;
            let (pending, xs, ys) = py.allow_threads(|| {
                train_streaming_with_counts(doc_words, connected, range_start, num_merges, sample_every)
            });
            self.merges.extend(pending);
            return Ok((xs, ys));
        }

        let graph = &mut self.graph;

        let (pending, xs, ys) = py.allow_threads(|| {
            let mut pending = Vec::new();
            let mut xs = vec![0usize];
            let mut ys = vec![graph.node_count()];

            if let GraphV::Unconn(subs_arc) = graph {
                let subs = Arc::make_mut(subs_arc);
                let mut active: Vec<bool> = subs
                    .iter()
                    .map(|sg| !matches!(sg, GraphV::Node(_)))
                    .collect();

                for i in range_start..num_merges {
                    match do_merge_step(subs, &mut active, false) {
                        Some((token, nodes)) => pending.push((token, nodes)),
                        None => break,
                    }

                    let merge_num = i + 1;
                    if merge_num % sample_every == 0 || merge_num == num_merges {
                        let count: usize = subs.iter().map(|sg| sg.node_count()).sum();
                        xs.push(merge_num);
                        ys.push(count);
                    }
                }
            } else {
                for i in range_start..num_merges {
                    let mut counts: FxHashMap<Vec<GraphV>, usize> = FxHashMap::default();
                    graph.emit_merges(&mut |m| *counts.entry(m).or_insert(0) += 1);
                    let Some((nodes, _)) = pick_best(&counts) else {
                        break;
                    };
                    let token = make_token(&nodes);
                    *graph = graph.merge(&token, &nodes);
                    apply_merge_to_cluster_cache(&token, &nodes);
                    pending.push((token, nodes));

                    let merge_num = i + 1;
                    if merge_num % sample_every == 0 || merge_num == num_merges {
                        xs.push(merge_num);
                        ys.push(graph.node_count());
                    }
                }
            }
            (pending, xs, ys)
        });

        self.merges.extend(pending);
        Ok((xs, ys))
    }

    fn make_trainer(&self) -> Self {
        Trainer {
            graph: self.graph.clone(),
            merges: self.merges.clone(),
            doc_words: self.doc_words.clone(),
            streaming_connected: self.streaming_connected,
        }
    }

    fn get_merges<'py>(&self, py: Python<'py>) -> PyResult<Vec<PyObject>> {
        self.merges
            .iter()
            .map(|(_, nodes)| {
                let strs: Vec<String> = nodes.iter().map(|n| n.to_str_repr()).collect();
                Ok(PyTuple::new(py, &strs)?.into())
            })
            .collect()
    }

    #[pyo3(name = "merges")]
    #[getter]
    fn py_merges<'py>(&self, py: Python<'py>) -> PyResult<Vec<PyObject>> {
        self.merges
            .iter()
            .map(|(token, nodes)| {
                let py_token = graphv_to_pyobject(py, token);
                let py_nodes: Vec<PyObject> =
                    nodes.iter().map(|n| graphv_to_pyobject(py, n)).collect();
                let nodes_tuple = PyTuple::new(py, &py_nodes)?;
                Ok(PyTuple::new(py, &[py_token, nodes_tuple.into()])?.into())
            })
            .collect()
    }

    #[getter]
    fn get_graph<'py>(&self, py: Python<'py>) -> PyObject {
        graphv_to_pyobject(py, &self.graph)
    }
    #[setter]
    fn set_graph(&mut self, graph: &Bound<'_, PyAny>) -> PyResult<()> {
        self.graph = pyobject_to_graphv(graph)?;
        Ok(())
    }

    #[getter]
    fn merges_raw<'py>(&self, py: Python<'py>) -> PyResult<Vec<PyObject>> {
        self.merges
            .iter()
            .map(|(token, nodes)| {
                let py_token = graphv_to_pyobject(py, token);
                let py_nodes: Vec<PyObject> =
                    nodes.iter().map(|n| graphv_to_pyobject(py, n)).collect();
                Ok(
                    PyTuple::new(py, &[py_token, PyTuple::new(py, &py_nodes)?.into()])?
                        .into(),
                )
            })
            .collect()
    }

    fn apply_merge(
        &mut self,
        token: &Bound<'_, PyAny>,
        nodes: &Bound<'_, PyTuple>,
    ) -> PyResult<()> {
        let token_g = pyobject_to_graphv(token)?;
        let merge_nodes: Vec<GraphV> = nodes
            .iter()
            .map(|item| pyobject_to_graphv(&item))
            .collect::<PyResult<_>>()?;
        self.graph = self.graph.merge(&token_g, &merge_nodes);
        self.merges.push((token_g, merge_nodes));
        Ok(())
    }
}
