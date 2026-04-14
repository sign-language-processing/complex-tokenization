use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::settings::{MAX_MERGE_SIZE, ONLY_MINIMAL_MERGES};
use std::sync::atomic::Ordering;


#[derive(Debug, Clone)]
pub enum GraphV {
    Node(Vec<u8>),
    Seq(Arc<Vec<GraphV>>),
    Tree {
        root: Box<GraphV>,
        children: Arc<Vec<GraphV>>,
    },
    FullConn(Arc<Vec<GraphV>>),
    Unconn(Arc<Vec<GraphV>>),
}

impl GraphV {
    pub fn new_seq(nodes: Vec<GraphV>) -> GraphV {
        GraphV::Seq(Arc::new(nodes))
    }

    pub fn new_tree(root: GraphV, children: Vec<GraphV>) -> GraphV {
        GraphV::Tree {
            root: Box::new(root),
            children: Arc::new(children),
        }
    }

    pub fn new_fullconn(nodes: Vec<GraphV>) -> GraphV {
        GraphV::FullConn(Arc::new(nodes))
    }
}

impl PartialEq for GraphV {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (GraphV::Node(a), GraphV::Node(b)) => a == b,
            (GraphV::Seq(a), GraphV::Seq(b)) => a == b,
            (GraphV::Tree { root: r1, children: c1 }, GraphV::Tree { root: r2, children: c2 }) => {
                r1 == r2 && c1 == c2
            }
            (GraphV::FullConn(a), GraphV::FullConn(b)) => a == b,
            (GraphV::Unconn(a), GraphV::Unconn(b)) => a == b,
            _ => false,
        }
    }
}
impl Eq for GraphV {}

impl Hash for GraphV {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            GraphV::Node(v) => v.hash(state),
            GraphV::Seq(nodes) => nodes.hash(state),
            GraphV::Tree { root, children } => {
                root.hash(state);
                children.hash(state);
            }
            GraphV::FullConn(nodes) => nodes.hash(state),
            GraphV::Unconn(subs) => subs.hash(state),
        }
    }
}

impl GraphV {
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            GraphV::Node(v) => v.clone(),
            GraphV::Seq(nodes) => {
                let mut buf = Vec::new();
                for n in nodes.as_ref() {
                    buf.extend(n.to_bytes());
                }
                buf
            }
            GraphV::Tree {
                root, children, ..
            } => {
                let mut buf = root.to_bytes();
                for c in children.as_ref() {
                    buf.extend(c.to_bytes());
                }
                buf
            }
            GraphV::FullConn(nodes) => {
                let mut buf = Vec::new();
                for n in nodes.as_ref() {
                    buf.extend(n.to_bytes());
                }
                buf
            }
            GraphV::Unconn(_) => panic!("Cannot convert UnconnectedGraphs to bytes"),
        }
    }

    pub fn to_str_repr(&self) -> String {
        let bytes = self.to_bytes();
        let s = String::from_utf8_lossy(&bytes).to_string();
        // Try IDS character lookup
        if let Some(ch) = crate::units::get_character_for_ids_str(&s) {
            return ch;
        }
        s
    }

    pub fn node_count(&self) -> usize {
        match self {
            GraphV::Node(_) => 1,
            GraphV::Seq(nodes) => nodes.iter().map(|n| n.node_count()).sum(),
            GraphV::Tree { root, children } => {
                root.node_count() + children.iter().map(|c| c.node_count()).sum::<usize>()
            }
            GraphV::FullConn(nodes) => nodes.iter().map(|n| n.node_count()).sum(),
            GraphV::Unconn(subs) => subs.iter().map(|sg| sg.node_count()).sum(),
        }
    }

    pub fn emit_merges(&self, emit: &mut dyn FnMut(Vec<GraphV>)) {
        match self {
            GraphV::Node(_) => {}
            GraphV::Seq(nodes) => seq_emit_merges(nodes, emit),
            GraphV::Tree { root, children } => tree_emit_merges(root, children, emit),
            GraphV::FullConn(nodes) => fullconn_emit_merges(nodes, emit),
            GraphV::Unconn(subs) => {
                for sg in subs.as_ref() {
                    sg.emit_merges(emit);
                }
            }
        }
    }

    pub fn get_merges(&self) -> Vec<Vec<GraphV>> {
        let mut result = Vec::new();
        self.emit_merges(&mut |m| result.push(m));
        result
    }

    pub fn merge(&self, token: &GraphV, merge: &[GraphV]) -> GraphV {
        match self {
            GraphV::Node(_) => self.clone(),
            GraphV::Seq(nodes) => seq_merge(nodes, token, merge),
            GraphV::Tree { root, children } => tree_merge(root, children, token, merge),
            GraphV::FullConn(nodes) => fullconn_merge(nodes, token, merge),
            GraphV::Unconn(subs) => {
                let new_subs: Vec<GraphV> = subs.iter().map(|sg| sg.merge(token, merge)).collect();
                if new_subs == **subs {
                    return self.clone();
                }
                GraphV::Unconn(Arc::new(new_subs))
            }
        }
    }

    /// Like merge but returns None when nothing changed — avoids allocations.
    pub fn try_merge(&self, token: &GraphV, merge: &[GraphV]) -> Option<GraphV> {
        match self {
            GraphV::Node(_) => None,
            GraphV::Seq(nodes) => seq_try_merge(nodes, token, merge),
            GraphV::Tree { root, children } => tree_try_merge(root, children, token, merge),
            GraphV::FullConn(nodes) => {
                let result = fullconn_merge(nodes, token, merge);
                if result == *self { None } else { Some(result) }
            }
            GraphV::Unconn(_) => {
                let result = self.merge(token, merge);
                if result == *self { None } else { Some(result) }
            }
        }
    }

    /// Check if any leaf Node in this graph has the given byte value.
    /// Zero-alloc, early-exit recursive search.
    pub fn contains_node(&self, target: &GraphV) -> bool {
        match self {
            GraphV::Node(_) => self == target,
            GraphV::Seq(nodes) => nodes.iter().any(|n| n.contains_node(target)),
            GraphV::Tree { root, children } => {
                root.contains_node(target) || children.iter().any(|c| c.contains_node(target))
            }
            GraphV::FullConn(nodes) => nodes.iter().any(|n| n.contains_node(target)),
            GraphV::Unconn(subs) => subs.iter().any(|sg| sg.contains_node(target)),
        }
    }

    pub fn is_node(&self) -> bool {
        matches!(self, GraphV::Node(_))
    }

    pub fn add(&self, other: &GraphV) -> GraphV {
        match (self, other) {
            (GraphV::Node(a), GraphV::Node(b)) => {
                let mut v = a.clone();
                v.extend(b);
                GraphV::Node(v)
            }
            (GraphV::Node(_), GraphV::Seq(nodes)) => {
                let mut new_nodes = vec![self.clone()];
                new_nodes.extend(nodes.as_ref().iter().cloned());
                GraphV::new_seq(new_nodes)
            }
            _ => {
                // NodesSequence + NodesSequence
                if let (GraphV::Seq(a), GraphV::Seq(b)) = (self, other) {
                    let mut new_nodes = a.as_ref().clone();
                    new_nodes.extend(b.as_ref().iter().cloned());
                    return GraphV::new_seq(new_nodes);
                }
                if let (GraphV::Seq(a), GraphV::Node(_)) = (self, other) {
                    let mut new_nodes = a.as_ref().clone();
                    new_nodes.push(other.clone());
                    return GraphV::new_seq(new_nodes);
                }
                panic!("unsupported add operation");
            }
        }
    }
}

fn seq_emit_merges(nodes: &[GraphV], emit: &mut dyn FnMut(Vec<GraphV>)) {
    let num_nodes = nodes.len();
    let only_minimal = ONLY_MINIMAL_MERGES.load(Ordering::Relaxed);
    let max_size = MAX_MERGE_SIZE.load(Ordering::Relaxed);

    // Fast path for BPE: only_minimal + max_size==2, all nodes are Node
    if only_minimal && max_size == 2 {
        let all_nodes = nodes.iter().all(|n| n.is_node());
        if all_nodes {
            for i in 0..num_nodes.saturating_sub(1) {
                emit(vec![nodes[i].clone(), nodes[i + 1].clone()]);
            }
            return;
        }
    }

    for i in 0..num_nodes {
        match &nodes[i] {
            GraphV::Node(_) => {}
            other => other.emit_merges(emit),
        }

        if only_minimal && !nodes[i].is_node() {
            continue;
        }

        let end = std::cmp::min(i + max_size + 1, num_nodes + 1);
        for j in (i + 2)..end {
            if only_minimal && !nodes[j - 1].is_node() {
                break;
            }
            if j - i == 2 {
                emit(vec![nodes[i].clone(), nodes[j - 1].clone()]);
            } else {
                emit(nodes[i..j].to_vec());
            }
        }
    }
}

fn seq_merge(nodes: &[GraphV], token: &GraphV, merge: &[GraphV]) -> GraphV {
    let m = merge.len();
    let n = nodes.len();
    let mut out: Vec<GraphV> = Vec::new();
    let mut i = 0;
    let mut found = false;

    while i + m <= n {
        if nodes[i..i + m] == *merge {
            out.push(token.clone());
            i += m;
            found = true;
        } else {
            out.push(nodes[i].clone());
            i += 1;
        }
    }
    while i < n {
        out.push(nodes[i].clone());
        i += 1;
    }

    if out.len() == 1 {
        let single = out.into_iter().next().unwrap();
        if found {
            return single;
        }
        return single.merge(token, merge);
    }

    // Recurse into children, tracking changes
    let mut changed = found;
    let merged: Vec<GraphV> = out
        .iter()
        .map(|n| {
            let new_n = n.merge(token, merge);
            if new_n != *n {
                changed = true;
            }
            new_n
        })
        .collect();

    if !changed {
        return GraphV::new_seq(nodes.to_vec());
    }
    GraphV::new_seq(merged)
}

fn tree_emit_merges(root: &GraphV, children: &[GraphV], emit: &mut dyn FnMut(Vec<GraphV>)) {
    root.emit_merges(emit);
    let only_minimal = ONLY_MINIMAL_MERGES.load(Ordering::Relaxed);
    if !only_minimal || (root.is_node() && children.iter().all(|c| c.is_node())) {
        let mut full_merge = vec![root.clone()];
        full_merge.extend(children.iter().cloned());
        emit(full_merge);
    }
    for child in children { child.emit_merges(emit); }
}

fn tree_merge(root: &GraphV, children: &[GraphV], token: &GraphV, merge: &[GraphV]) -> GraphV {
    if merge.len() == children.len() + 1 && merge[0] == *root {
        if merge[1..].iter().zip(children.iter()).all(|(a, b)| a == b) {
            return GraphV::Node(token.to_bytes());
        }
    }

    let new_root = root.merge(token, merge);
    let mut changed = new_root != *root;
    let new_children: Vec<GraphV> = children
        .iter()
        .map(|c| {
            let new_c = c.merge(token, merge);
            if new_c != *c {
                changed = true;
            }
            new_c
        })
        .collect();

    if !changed {
        return GraphV::new_tree(root.clone(), children.to_vec());
    }
    GraphV::new_tree(new_root, new_children)
}

fn fullconn_emit_merges(nodes: &[GraphV], emit: &mut dyn FnMut(Vec<GraphV>)) {
    for node in nodes { node.emit_merges(emit); }
    for i in 0..nodes.len() {
        for j in 0..nodes.len() {
            if i != j { emit(vec![nodes[i].clone(), nodes[j].clone()]); }
        }
    }
}

fn fullconn_merge(nodes: &[GraphV], token: &GraphV, merge: &[GraphV]) -> GraphV {
    let remaining: Vec<GraphV> = nodes.to_vec();
    if merge.len() == 2 {
        let m0 = &merge[0];
        let m1 = &merge[1];
        for i in 0..remaining.len() {
            if remaining[i] == *m0 {
                for j in 0..remaining.len() {
                    if i != j && remaining[j] == *m1 {
                        let mut merged: Vec<GraphV> = remaining
                            .iter()
                            .enumerate()
                            .filter(|&(k, _)| k != i && k != j)
                            .map(|(_, n)| n.clone())
                            .collect();
                        merged.push(token.clone());
                        if merged.len() == 1 {
                            return merged.into_iter().next().unwrap();
                        }
                        return GraphV::new_fullconn(merged);
                    }
                }
            }
        }
    }

    let new_nodes: Vec<GraphV> = nodes.iter().map(|n| n.merge(token, merge)).collect();
    if new_nodes == nodes {
        return GraphV::new_fullconn(nodes.to_vec());
    }
    GraphV::new_fullconn(new_nodes)
}

// --- Python wrapper types ---

pub fn graphv_to_pyobject(py: Python<'_>, g: &GraphV) -> PyObject {
    match g {
        GraphV::Node(_) => Node { inner: g.clone() }.into_pyobject(py).unwrap().into_any().unbind(),
        GraphV::Seq(..) => NodesSequence { inner: g.clone() }.into_pyobject(py).unwrap().into_any().unbind(),
        GraphV::Tree { .. } => Tree { inner: g.clone() }.into_pyobject(py).unwrap().into_any().unbind(),
        GraphV::FullConn(..) => FullyConnectedGraph { inner: g.clone() }.into_pyobject(py).unwrap().into_any().unbind(),
        GraphV::Unconn(_) => UnconnectedGraphs { inner: g.clone() }.into_pyobject(py).unwrap().into_any().unbind(),
    }
}

pub fn pyobject_to_graphv(obj: &Bound<'_, PyAny>) -> PyResult<GraphV> {
    if let Ok(n) = obj.extract::<Node>() {
        return Ok(n.inner);
    }
    if let Ok(n) = obj.extract::<NodesSequence>() {
        return Ok(n.inner);
    }
    if let Ok(n) = obj.extract::<Tree>() {
        return Ok(n.inner);
    }
    if let Ok(n) = obj.extract::<FullyConnectedGraph>() {
        return Ok(n.inner);
    }
    if let Ok(n) = obj.extract::<UnconnectedGraphs>() {
        return Ok(n.inner);
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "Expected a GraphVertex type (Node, NodesSequence, Tree, FullyConnectedGraph, UnconnectedGraphs)",
    ))
}

fn merge_tuple_to_py(py: Python<'_>, merge: &[GraphV]) -> PyObject {
    let items: Vec<PyObject> = merge.iter().map(|g| graphv_to_pyobject(py, g)).collect();
    PyTuple::new(py, &items).unwrap().into()
}

// --- Node ---

#[pyclass(frozen, eq)]
#[derive(Clone, Debug)]
pub struct Node {
    pub inner: GraphV,
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

#[pymethods]
impl Node {
    #[new]
    fn new(value: Vec<u8>) -> Self {
        Node { inner: GraphV::Node(value) }
    }

    #[getter]
    fn value(&self) -> &[u8] {
        match &self.inner {
            GraphV::Node(v) => v,
            _ => unreachable!(),
        }
    }

    fn __bytes__(&self) -> Vec<u8> {
        self.inner.to_bytes()
    }

    fn __str__(&self) -> String {
        self.inner.to_str_repr()
    }

    fn __repr__(&self) -> String {
        format!("Node(value={:?})", self.value())
    }

    fn __len__(&self) -> usize {
        self.value().len()
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let other_g = pyobject_to_graphv(other)?;
        let result = self.inner.add(&other_g);
        Python::with_gil(|py| Ok(graphv_to_pyobject(py, &result)))
    }

    fn merge(&self, _token: &Node, _merge: &Bound<'_, PyTuple>) -> Self {
        self.clone()
    }

    fn get_merges(&self) -> Vec<Vec<Node>> {
        Vec::new()
    }

    fn dot(&self, level: Option<usize>) -> Vec<String> {
        let level = level.unwrap_or(0);
        let label = dot_escape(&self.inner.to_str_repr());
        let oid = format!("o{:x}", std::ptr::from_ref(self) as usize);
        vec![format!(
            "{}{}",
            "\t".repeat(level),
            format!("{oid} [label=\"{label}\"];")
        )]
    }

    fn node_count(&self) -> usize { self.inner.node_count() }

    fn oid(&self) -> String {
        format!("o{:x}", std::ptr::from_ref(self) as usize)
    }
}

// --- NodesSequence ---

#[pyclass(frozen, eq)]
#[derive(Clone, Debug)]
pub struct NodesSequence {
    pub inner: GraphV,
}

impl PartialEq for NodesSequence {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl NodesSequence {
    fn nodes_ref(&self) -> &[GraphV] {
        match &self.inner {
            GraphV::Seq(nodes) => nodes,
            _ => unreachable!(),
        }
    }
}

#[pymethods]
impl NodesSequence {
    #[new]
    fn new(nodes: &Bound<'_, PyTuple>) -> PyResult<Self> {
        let mut inner_nodes = Vec::with_capacity(nodes.len());
        for item in nodes.iter() {
            inner_nodes.push(pyobject_to_graphv(&item)?);
        }
        Ok(NodesSequence {
            inner: GraphV::new_seq(inner_nodes),
        })
    }

    #[getter]
    fn nodes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let nodes = self.nodes_ref();
        let items: Vec<PyObject> = nodes.iter().map(|g| graphv_to_pyobject(py, g)).collect();
        PyTuple::new(py, &items)
    }

    fn __bytes__(&self) -> Vec<u8> {
        self.inner.to_bytes()
    }

    fn __str__(&self) -> String {
        self.inner.to_str_repr()
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let other_g = pyobject_to_graphv(other)?;
        let result = self.inner.add(&other_g);
        Python::with_gil(|py| Ok(graphv_to_pyobject(py, &result)))
    }

    fn merge<'py>(&self, py: Python<'py>, token: &Node, merge_tuple: &Bound<'py, PyTuple>) -> PyResult<PyObject> {
        let merge = py_tuple_to_merge(merge_tuple)?;
        let result = self.inner.merge(&token.inner, &merge);
        Ok(graphv_to_pyobject(py, &result))
    }

    fn get_merges<'py>(&self, py: Python<'py>) -> PyResult<Vec<PyObject>> {
        let merges = self.inner.get_merges();
        Ok(merges.iter().map(|m| merge_tuple_to_py(py, m)).collect())
    }

    fn node_count(&self) -> usize { self.inner.node_count() }

    fn oid(&self) -> String {
        format!("o{:x}", std::ptr::from_ref(self) as usize)
    }

    fn dot(&self, _level: Option<usize>) -> Vec<String> {
        vec!["/* NodesSequence dot not implemented in Rust */".to_string()]
    }
}

// --- Tree ---

#[pyclass(frozen, eq)]
#[derive(Clone, Debug)]
pub struct Tree {
    pub inner: GraphV,
}

impl PartialEq for Tree {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl Tree {
    fn root_ref(&self) -> &GraphV {
        match &self.inner {
            GraphV::Tree { root, .. } => root,
            _ => unreachable!(),
        }
    }

    fn children_ref(&self) -> &[GraphV] {
        match &self.inner {
            GraphV::Tree { children, .. } => children,
            _ => unreachable!(),
        }
    }
}

#[pymethods]
impl Tree {
    #[new]
    fn new(root: &Bound<'_, PyAny>, children: &Bound<'_, PyTuple>) -> PyResult<Self> {
        let root_g = pyobject_to_graphv(root)?;
        let mut child_gs = Vec::with_capacity(children.len());
        for item in children.iter() {
            child_gs.push(pyobject_to_graphv(&item)?);
        }
        Ok(Tree {
            inner: GraphV::new_tree(root_g, child_gs),
        })
    }

    #[getter]
    fn root<'py>(&self, py: Python<'py>) -> PyObject {
        graphv_to_pyobject(py, self.root_ref())
    }

    #[getter]
    fn children<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let children = self.children_ref();
        let items: Vec<PyObject> = children.iter().map(|g| graphv_to_pyobject(py, g)).collect();
        PyTuple::new(py, &items)
    }

    fn __bytes__(&self) -> Vec<u8> {
        self.inner.to_bytes()
    }

    fn __str__(&self) -> String {
        self.inner.to_str_repr()
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }

    fn merge<'py>(&self, py: Python<'py>, token: &Node, merge_tuple: &Bound<'py, PyTuple>) -> PyResult<PyObject> {
        let merge = py_tuple_to_merge(merge_tuple)?;
        let result = self.inner.merge(&token.inner, &merge);
        Ok(graphv_to_pyobject(py, &result))
    }

    fn get_merges<'py>(&self, py: Python<'py>) -> PyResult<Vec<PyObject>> {
        let merges = self.inner.get_merges();
        Ok(merges.iter().map(|m| merge_tuple_to_py(py, m)).collect())
    }

    fn node_count(&self) -> usize { self.inner.node_count() }

    fn oid(&self) -> String {
        format!("o{:x}", std::ptr::from_ref(self) as usize)
    }

    fn dot(&self, _level: Option<usize>) -> Vec<String> {
        vec!["/* Tree dot not implemented in Rust */".to_string()]
    }
}

// --- FullyConnectedGraph ---

#[pyclass(frozen, eq)]
#[derive(Clone, Debug)]
pub struct FullyConnectedGraph {
    pub inner: GraphV,
}

impl PartialEq for FullyConnectedGraph {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl FullyConnectedGraph {
    fn nodes_ref(&self) -> &[GraphV] {
        match &self.inner {
            GraphV::FullConn(nodes) => nodes,
            _ => unreachable!(),
        }
    }
}

#[pymethods]
impl FullyConnectedGraph {
    #[new]
    fn new(nodes: &Bound<'_, PyTuple>) -> PyResult<Self> {
        let mut inner_nodes = Vec::with_capacity(nodes.len());
        for item in nodes.iter() {
            inner_nodes.push(pyobject_to_graphv(&item)?);
        }
        Ok(FullyConnectedGraph {
            inner: GraphV::new_fullconn(inner_nodes),
        })
    }

    #[getter]
    fn nodes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let nodes = self.nodes_ref();
        let items: Vec<PyObject> = nodes.iter().map(|g| graphv_to_pyobject(py, g)).collect();
        PyTuple::new(py, &items)
    }

    fn __bytes__(&self) -> Vec<u8> {
        self.inner.to_bytes()
    }

    fn __str__(&self) -> String {
        self.inner.to_str_repr()
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }

    fn merge<'py>(&self, py: Python<'py>, token: &Node, merge_tuple: &Bound<'py, PyTuple>) -> PyResult<PyObject> {
        let merge = py_tuple_to_merge(merge_tuple)?;
        let result = self.inner.merge(&token.inner, &merge);
        Ok(graphv_to_pyobject(py, &result))
    }

    fn get_merges<'py>(&self, py: Python<'py>) -> PyResult<Vec<PyObject>> {
        let merges = self.inner.get_merges();
        Ok(merges.iter().map(|m| merge_tuple_to_py(py, m)).collect())
    }

    fn node_count(&self) -> usize { self.inner.node_count() }

    fn oid(&self) -> String {
        format!("o{:x}", std::ptr::from_ref(self) as usize)
    }

    fn dot(&self, _level: Option<usize>) -> Vec<String> {
        vec!["/* FullyConnectedGraph dot not implemented in Rust */".to_string()]
    }
}

// --- UnconnectedGraphs ---

#[pyclass(frozen, eq)]
#[derive(Clone, Debug)]
pub struct UnconnectedGraphs {
    pub inner: GraphV,
}

impl PartialEq for UnconnectedGraphs {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl UnconnectedGraphs {
    fn subgraphs_ref(&self) -> &[GraphV] {
        match &self.inner {
            GraphV::Unconn(subs) => subs,
            _ => unreachable!(),
        }
    }
}

#[pymethods]
impl UnconnectedGraphs {
    #[new]
    fn new(subgraphs: &Bound<'_, PyTuple>) -> PyResult<Self> {
        let mut inner_subs = Vec::with_capacity(subgraphs.len());
        for item in subgraphs.iter() {
            inner_subs.push(pyobject_to_graphv(&item)?);
        }
        Ok(UnconnectedGraphs {
            inner: GraphV::Unconn(Arc::new(inner_subs)),
        })
    }

    #[getter]
    fn subgraphs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let subs = self.subgraphs_ref();
        let items: Vec<PyObject> = subs.iter().map(|g| graphv_to_pyobject(py, g)).collect();
        PyTuple::new(py, &items)
    }

    fn __bytes__(&self) -> PyResult<Vec<u8>> {
        Err(PyErr::new::<pyo3::exceptions::PyException, _>(
            "Cannot convert UnconnectedGraphs to bytes",
        ))
    }

    fn __str__(&self) -> String {
        "UnconnectedGraphs(...)".to_string()
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }

    fn merge<'py>(&self, py: Python<'py>, token: &Node, merge_tuple: &Bound<'py, PyTuple>) -> PyResult<PyObject> {
        let merge = py_tuple_to_merge(merge_tuple)?;
        let result = self.inner.merge(&token.inner, &merge);
        Ok(graphv_to_pyobject(py, &result))
    }

    fn get_merges<'py>(&self, py: Python<'py>) -> PyResult<Vec<PyObject>> {
        let merges = self.inner.get_merges();
        Ok(merges.iter().map(|m| merge_tuple_to_py(py, m)).collect())
    }

    fn node_count(&self) -> usize { self.inner.node_count() }

    fn oid(&self) -> String {
        format!("o{:x}", std::ptr::from_ref(self) as usize)
    }

    fn dot(&self, _level: Option<usize>) -> Vec<String> {
        vec!["/* UnconnectedGraphs dot not implemented in Rust */".to_string()]
    }
}

// --- Helpers ---

fn py_tuple_to_merge(tuple: &Bound<'_, PyTuple>) -> PyResult<Vec<GraphV>> {
    let mut merge = Vec::with_capacity(tuple.len());
    for item in tuple.iter() {
        merge.push(pyobject_to_graphv(&item)?);
    }
    Ok(merge)
}

fn seq_try_merge(nodes: &[GraphV], token: &GraphV, merge: &[GraphV]) -> Option<GraphV> {
    let m = merge.len();
    let n = nodes.len();

    let only_minimal = ONLY_MINIMAL_MERGES.load(Ordering::Relaxed);

    if only_minimal && m == 2 && n >= 2 {
        return seq_try_merge_minimal_2(nodes, token, &merge[0], &merge[1]);
    }

    let has_complex_child = nodes.iter().any(|n| !matches!(n, GraphV::Node(_)));

    let has_match = if n < m || (!has_complex_child && only_minimal) {
        false
    } else {
        let mut found = false;
        let mut i = 0;
        while i + m <= n {
            if nodes[i..i + m] == *merge {
                found = true;
                break;
            }
            i += 1;
        }
        found
    };

    if !has_match && !has_complex_child {
        return None;
    }

    let mut out: Vec<GraphV> = Vec::with_capacity(n);
    let mut i = 0;
    while i + m <= n {
        if nodes[i..i + m] == *merge {
            out.push(token.clone());
            i += m;
        } else {
            out.push(nodes[i].clone());
            i += 1;
        }
    }
    while i < n {
        out.push(nodes[i].clone());
        i += 1;
    }

    if out.len() == 1 {
        let single = out.into_iter().next().unwrap();
        if has_match { return Some(single); }
        return single.try_merge(token, merge);
    }

    let mut child_changed = false;
    let mut new_out: Vec<GraphV> = Vec::with_capacity(out.len());
    for node in &out {
        if matches!(node, GraphV::Node(_)) {
            new_out.push(node.clone());
        } else {
            match node.try_merge(token, merge) {
                Some(new_n) => { child_changed = true; new_out.push(new_n); }
                None => new_out.push(node.clone()),
            }
        }
    }
    if !has_match && !child_changed { return None; }
    Some(GraphV::new_seq(new_out))
}

fn seq_try_merge_minimal_2(nodes: &[GraphV], token: &GraphV, m0: &GraphV, m1: &GraphV) -> Option<GraphV> {
    let n = nodes.len();
    if n < 2 { return None; }

    let mut first_match: usize = n;
    let mut has_complex = false;

    for i in 0..n {
        if !matches!(&nodes[i], GraphV::Node(_)) {
            has_complex = true;
        }
        if first_match == n && i + 1 < n && nodes[i] == *m0 && nodes[i + 1] == *m1 {
            first_match = i;
        }
        if first_match < n && has_complex { break; }
    }

    if first_match == n && !has_complex {
        return None;
    }

    if first_match < n {
        let mut out: Vec<GraphV> = Vec::with_capacity(n);
        for j in 0..first_match {
            out.push(nodes[j].clone());
        }
        out.push(token.clone());
        let mut i = first_match + 2;
        while i + 1 < n {
            if nodes[i] == *m0 && nodes[i + 1] == *m1 {
                out.push(token.clone());
                i += 2;
            } else {
                out.push(nodes[i].clone());
                i += 1;
            }
        }
        if i < n { out.push(nodes[i].clone()); }

        if out.len() == 1 {
            return Some(out.into_iter().next().unwrap());
        }
        if !has_complex {
            return Some(GraphV::new_seq(out));
        }

        let merge = [m0.clone(), m1.clone()];
        let new_out: Vec<GraphV> = out.into_iter().map(|node| {
            if matches!(&node, GraphV::Node(_)) { return node; }
            match node.try_merge(token, &merge) {
                Some(new_n) => new_n,
                None => node,
            }
        }).collect();
        return Some(GraphV::new_seq(new_out));
    }

    let merge = [m0.clone(), m1.clone()];
    let mut child_changed = false;
    let new_nodes: Vec<GraphV> = nodes.iter().map(|node| {
        if matches!(node, GraphV::Node(_)) { return node.clone(); }
        match node.try_merge(token, &merge) {
            Some(new_n) => { child_changed = true; new_n }
            None => node.clone(),
        }
    }).collect();
    if !child_changed { return None; }
    Some(GraphV::new_seq(new_nodes))
}

fn tree_try_merge(root: &GraphV, children: &[GraphV], token: &GraphV, merge: &[GraphV]) -> Option<GraphV> {
    // Check if full tree merge (root + all children match)
    if merge.len() == children.len() + 1 && merge[0] == *root {
        if merge[1..].iter().zip(children.iter()).all(|(a, b)| a == b) {
            return Some(GraphV::Node(token.to_bytes()));
        }
    }

    let new_root = root.try_merge(token, merge);
    let new_children: Vec<Option<GraphV>> = children.iter()
        .map(|c| c.try_merge(token, merge))
        .collect();

    let any_changed = new_root.is_some() || new_children.iter().any(|c| c.is_some());
    if !any_changed { return None; }

    Some(GraphV::new_tree(
        new_root.unwrap_or_else(|| root.clone()),
        children
            .iter()
            .zip(new_children.into_iter())
            .map(|(old, new)| new.unwrap_or_else(|| old.clone()))
            .collect(),
    ))
}

fn dot_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

impl fmt::Display for GraphV {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_str_repr())
    }
}
