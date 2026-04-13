use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::sync::Arc;

use crate::graph::{graphv_to_pyobject, pyobject_to_graphv, GraphV};

/// Counts merge candidates while preserving first-seen insertion order.
/// Matches the behavior of Python's `Counter(graph.get_merges())`.
struct OrderedCounter {
    index: FxHashMap<Vec<GraphV>, usize>,
    keys: Vec<Vec<GraphV>>,
    counts: Vec<usize>,
}

impl OrderedCounter {
    fn new() -> Self {
        OrderedCounter {
            index: FxHashMap::default(),
            keys: Vec::new(),
            counts: Vec::new(),
        }
    }

    fn add(&mut self, merge: Vec<GraphV>) {
        if let Some(&idx) = self.index.get(&merge) {
            self.counts[idx] += 1;
        } else {
            self.index.insert(merge.clone(), self.keys.len());
            self.keys.push(merge);
            self.counts.push(1);
        }
    }

    fn merge_from(&mut self, other: Self) {
        for (i, key) in other.keys.into_iter().enumerate() {
            if let Some(&idx) = self.index.get(&key) {
                self.counts[idx] += other.counts[i];
            } else {
                self.index.insert(key.clone(), self.keys.len());
                self.keys.push(key);
                self.counts.push(other.counts[i]);
            }
        }
    }

    /// Find the merge with the highest score: (len - 1) * count.
    /// Returns the first one with the max score (matching Python's `max()`).
    fn best(&self) -> Option<usize> {
        if self.keys.is_empty() {
            return None;
        }
        let mut best_idx = 0;
        let mut best_score = 0usize;
        for (i, key) in self.keys.iter().enumerate() {
            let score = (key.len() - 1) * self.counts[i];
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }
        if best_score == 0 {
            return None;
        }
        Some(best_idx)
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

/// Count merge candidates from subgraphs with parallel chunking,
/// preserving first-seen order across all subgraphs.
fn count_merges_ordered(subs: &[GraphV], active: &[bool]) -> OrderedCounter {
    let n_threads = rayon::current_num_threads().max(1);
    let chunk_size = (subs.len() / n_threads).max(1);

    let chunk_counters: Vec<OrderedCounter> = subs
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            let mut counter = OrderedCounter::new();
            let base = chunk_idx * chunk_size;
            for (offset, sg) in chunk.iter().enumerate() {
                if active[base + offset] {
                    sg.emit_merges(&mut |m| counter.add(m));
                }
            }
            counter
        })
        .collect();

    let mut global = OrderedCounter::new();
    for counter in chunk_counters {
        global.merge_from(counter);
    }
    global
}

/// Apply merge to subgraphs in parallel using try_merge.
fn apply_merge(subs: &mut [GraphV], active: &mut [bool], token: &GraphV, merge: &[GraphV]) {
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
        let counter = count_merges_ordered(subs, &active);
        let Some(best_idx) = counter.best() else {
            break;
        };
        let nodes = counter.keys[best_idx].clone();
        let token = nodes
            .iter()
            .skip(1)
            .fold(nodes[0].clone(), |acc, n| acc.add(n));
        if verbose {
            println!("Merging {:?} count={}", nodes, counter.counts[best_idx]);
        }
        apply_merge(subs, &mut active, &token, &nodes);
        pending.push((token, nodes));
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
        let mut counter = OrderedCounter::new();
        graph.emit_merges(&mut |m| counter.add(m));
        let Some(best_idx) = counter.best() else {
            break;
        };
        let nodes = counter.keys[best_idx].clone();
        let token = nodes
            .iter()
            .skip(1)
            .fold(nodes[0].clone(), |acc, n| acc.add(n));
        if verbose {
            println!("Merging {:?} count={}", nodes, counter.counts[best_idx]);
        }
        *graph = graph.merge(&token, &nodes);
        pending.push((token, nodes));
    }
    pending
}

#[pyclass]
pub struct Trainer {
    pub graph: GraphV,
    pub merges: Vec<(GraphV, Vec<GraphV>)>,
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
            }),
            (None, Some(gs)) => {
                let subs: Vec<GraphV> = gs
                    .iter()
                    .map(|item| pyobject_to_graphv(&item))
                    .collect::<PyResult<_>>()?;
                Ok(Trainer {
                    graph: flatten_unconn(GraphV::Unconn(Arc::new(subs))),
                    merges: Vec::new(),
                })
            }
        }
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
                    let counter = count_merges_ordered(subs, &active);
                    let Some(best_idx) = counter.best() else {
                        break;
                    };
                    let nodes = counter.keys[best_idx].clone();
                    let token = nodes
                        .iter()
                        .skip(1)
                        .fold(nodes[0].clone(), |acc, n| acc.add(n));
                    apply_merge(subs, &mut active, &token, &nodes);
                    pending.push((token, nodes));

                    let merge_num = i + 1;
                    if merge_num % sample_every == 0 || merge_num == num_merges {
                        let count: usize = subs.iter().map(|sg| sg.node_count()).sum();
                        xs.push(merge_num);
                        ys.push(count);
                    }
                }
            } else {
                for i in range_start..num_merges {
                    let mut counter = OrderedCounter::new();
                    graph.emit_merges(&mut |m| counter.add(m));
                    let Some(best_idx) = counter.best() else {
                        break;
                    };
                    let nodes = counter.keys[best_idx].clone();
                    let token = nodes
                        .iter()
                        .skip(1)
                        .fold(nodes[0].clone(), |acc, n| acc.add(n));
                    *graph = graph.merge(&token, &nodes);
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
