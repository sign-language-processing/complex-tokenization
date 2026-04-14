use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Mutex;
use unicode_segmentation::UnicodeSegmentation;

use crate::graph::{graphv_to_pyobject, pyobject_to_graphv, GraphV};

use std::sync::{Arc, LazyLock};

static CLUSTER_HANDLERS: LazyLock<Mutex<Vec<(String, regex::Regex, PyObject)>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));

static CLUSTER_CACHE: LazyLock<Mutex<HashMap<String, GraphV>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

// IDS reverse dictionary — loaded lazily from Python side
static IDS_REVERSE_DICT: LazyLock<Mutex<Option<HashMap<String, String>>>> =
    LazyLock::new(|| Mutex::new(None));

pub fn get_character_for_ids_str(ids: &str) -> Option<String> {
    let dict = IDS_REVERSE_DICT.lock().unwrap();
    dict.as_ref()?.get(ids).cloned()
}

pub fn set_ids_reverse_dict(dict: HashMap<String, String>) {
    let mut d = IDS_REVERSE_DICT.lock().unwrap();
    *d = Some(dict);
}

#[pyfunction]
pub fn set_ids_reverse_dict_py(dict: HashMap<String, String>) {
    set_ids_reverse_dict(dict);
}

#[pyfunction]
pub fn register_script(py: Python<'_>, script: String, handler: PyObject) -> PyResult<()> {
    let pattern = format!(r"\p{{{script}}}");
    let re = regex::Regex::new(&pattern).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid script name '{script}': {e}"))
    })?;
    let mut handlers = CLUSTER_HANDLERS.lock().unwrap();
    handlers.retain(|(s, _, _)| s != &script);
    handlers.push((script, re, handler.clone_ref(py)));
    // Invalidate cache when handlers change
    CLUSTER_CACHE.lock().unwrap().clear();
    Ok(())
}

#[pyfunction]
pub fn clear_handlers() {
    CLUSTER_HANDLERS.lock().unwrap().clear();
    CLUSTER_CACHE.lock().unwrap().clear();
}

pub fn apply_merge_to_cluster_cache(token: &GraphV, merge: &[GraphV]) {
    let mut cache = CLUSTER_CACHE.lock().unwrap();
    for graph in cache.values_mut() {
        if let Some(new_g) = graph.try_merge(token, merge) {
            *graph = new_g;
        }
    }
}

static WORD_CACHE: LazyLock<Mutex<HashMap<String, GraphV>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

pub fn snapshot_word_cache() -> HashMap<String, GraphV> {
    WORD_CACHE.lock().unwrap().clone()
}

pub fn replace_word_cache(new_cache: HashMap<String, GraphV>) {
    let mut cache = WORD_CACHE.lock().unwrap();
    *cache = new_cache;
}

pub fn warm_word_cache(py: Python<'_>, word: &str) -> PyResult<()> {
    {
        let cache = WORD_CACHE.lock().unwrap();
        if cache.contains_key(word) {
            return Ok(());
        }
    }
    let g = utf8_clusters_inner(py, word)?;
    WORD_CACHE.lock().unwrap().insert(word.to_string(), g);
    Ok(())
}

#[pyfunction]
pub fn warm_word_cache_py(py: Python<'_>, words: Vec<String>) -> PyResult<()> {
    for w in &words {
        warm_word_cache(py, w)?;
    }
    Ok(())
}

#[pyfunction]
pub fn clear_word_cache() {
    WORD_CACHE.lock().unwrap().clear();
}

#[pyfunction]
pub fn get_handlers_dict(py: Python<'_>) -> PyResult<PyObject> {
    let handlers = CLUSTER_HANDLERS.lock().unwrap();
    let dict = pyo3::types::PyDict::new(py);
    for (script, _, handler) in handlers.iter() {
        dict.set_item(script, handler.clone_ref(py))?;
    }
    Ok(dict.into())
}

fn get_handler<'a>(cluster: &str, handlers: &'a [(String, regex::Regex, PyObject)]) -> Option<&'a PyObject> {
    if handlers.is_empty() {
        return None;
    }
    let first_char: String = cluster.graphemes(true).next()?.chars().next()?.to_string();
    for (_, re, handler) in handlers {
        if re.is_match(&first_char) {
            return Some(handler);
        }
    }
    None
}

pub fn utf8_inner(s: &str) -> GraphV {
    let bytes: Vec<u8> = s.as_bytes().to_vec();
    if bytes.len() == 1 {
        return GraphV::Node(vec![bytes[0]]);
    }
    let nodes: Vec<GraphV> = bytes.iter().map(|&b| GraphV::Node(vec![b])).collect();
    GraphV::new_seq(nodes)
}

pub fn characters_inner(s: &str) -> GraphV {
    let chars: Vec<String> = s.chars().map(|c| c.to_string()).collect();
    if chars.len() == 1 {
        return GraphV::Node(chars[0].as_bytes().to_vec());
    }
    let nodes: Vec<GraphV> = chars
        .iter()
        .map(|c| GraphV::Node(c.as_bytes().to_vec()))
        .collect();
    GraphV::new_seq(nodes)
}

fn resolve_cluster(py: Python<'_>, cluster: &str, handlers: &[(String, regex::Regex, PyObject)]) -> PyResult<GraphV> {
    // Check cache first
    {
        let cache = CLUSTER_CACHE.lock().unwrap();
        if let Some(cached) = cache.get(cluster) {
            return Ok(cached.clone());
        }
    }

    let g = if let Some(handler) = get_handler(cluster, handlers) {
        let result = handler.call1(py, (cluster,))?;
        pyobject_to_graphv(result.bind(py))?
    } else {
        let chars: Vec<char> = cluster.chars().collect();
        if chars.len() == 1 {
            utf8_inner(cluster)
        } else {
            let char_nodes: Vec<GraphV> = chars
                .iter()
                .map(|c| utf8_inner(&c.to_string()))
                .collect();
            GraphV::new_seq(char_nodes)
        }
    };

    // Store in cache
    CLUSTER_CACHE.lock().unwrap().insert(cluster.to_string(), g.clone());

    Ok(g)
}

pub fn utf8_clusters_inner(py: Python<'_>, s: &str) -> PyResult<GraphV> {
    let clusters: Vec<&str> = s.graphemes(true).collect();
    let handlers = CLUSTER_HANDLERS.lock().unwrap();
    let mut nodes: Vec<GraphV> = Vec::with_capacity(clusters.len());

    for cluster in &clusters {
        nodes.push(resolve_cluster(py, cluster, &handlers)?);
    }

    if nodes.len() == 1 {
        return Ok(nodes.into_iter().next().unwrap());
    }
    Ok(GraphV::new_seq(nodes))
}

#[pyfunction]
pub fn utf8(py: Python<'_>, s: &str) -> PyResult<PyObject> {
    let g = utf8_inner(s);
    Ok(graphv_to_pyobject(py, &g))
}

#[pyfunction]
pub fn characters(py: Python<'_>, s: &str) -> PyResult<PyObject> {
    let g = characters_inner(s);
    Ok(graphv_to_pyobject(py, &g))
}

#[pyfunction]
pub fn utf8_clusters(py: Python<'_>, s: &str) -> PyResult<PyObject> {
    let g = utf8_clusters_inner(py, s)?;
    Ok(graphv_to_pyobject(py, &g))
}
