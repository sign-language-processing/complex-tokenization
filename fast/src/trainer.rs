use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::Arc;

use crate::graph::{graphv_to_pyobject, pyobject_to_graphv, GraphV};
use crate::units::{apply_merge_to_cluster_cache, replace_word_cache, snapshot_word_cache};
use std::collections::HashMap;

// Merge score: merging a k-tuple with `count` occurrences removes (k-1)*count nodes.
fn merge_score(nodes: &[GraphV], count: usize) -> usize {
    (nodes.len() - 1) * count
}

/// One pass over `map`: returns a representative max-score candidate, whether
/// the max score is shared by more than one candidate (a tie), and that score.
/// None when every candidate scores 0. The common no-tie path avoids allocating
/// a tie set — only ties pay for `tied_at`.
fn best_and_tie(map: &FxHashMap<Vec<GraphV>, usize>) -> Option<(&Vec<GraphV>, bool, usize)> {
    let mut best_score = 0;
    let mut best_key: Option<&Vec<GraphV>> = None;
    let mut tie = false;
    for (k, &c) in map {
        let s = merge_score(k, c);
        if s == 0 {
            continue;
        }
        if s > best_score {
            best_score = s;
            best_key = Some(k);
            tie = false;
        } else if s == best_score {
            tie = true;
        }
    }
    best_key.map(|k| (k, tie, best_score))
}

/// Candidates whose score equals `score`.
fn tied_at(map: &FxHashMap<Vec<GraphV>, usize>, score: usize) -> FxHashSet<Vec<GraphV>> {
    map.iter()
        .filter(|(k, &c)| merge_score(k, c) == score)
        .map(|(k, _)| k.clone())
        .collect()
}

/// First candidate of `graph` in emit (traversal) order that is in `ties`.
/// Mirrors the reference tie-break: among equal-max-score candidates, take the
/// one encountered first while walking the graph (== HuggingFace merge order).
fn first_in_ties(graph: &GraphV, ties: &FxHashSet<Vec<GraphV>>) -> Option<Vec<GraphV>> {
    crate::graph::first_merge_in(graph, ties)
}

/// Pick the best merge from sharded count maps, breaking ties by first
/// emit-order appearance while scanning `graphs` in order (the reference's
/// traversal order).
fn pick_best_by_scan<'a>(
    shards: &[FxHashMap<Vec<GraphV>, usize>],
    graphs: impl Iterator<Item = &'a GraphV>,
) -> Option<Vec<GraphV>> {
    let (best_key, tie, best) = best_and_tie_sharded(shards)?;
    if !tie {
        return Some(best_key.clone());
    }
    let mut ties: FxHashSet<Vec<GraphV>> = FxHashSet::default();
    for shard in shards {
        ties.extend(tied_at(shard, best));
    }
    let mut graphs = graphs;
    graphs.find_map(|g| first_in_ties(g, &ties))
}

/// Tie-break for the connected streaming path: scan documents in order, each
/// assembled on the fly, and take the first tied candidate.
fn pick_best_docs(
    map: &FxHashMap<Vec<GraphV>, usize>,
    doc_words: &[Vec<String>],
    cache: &HashMap<String, GraphV>,
) -> Option<Vec<GraphV>> {
    let (best_key, tie, best) = best_and_tie(map)?;
    if !tie {
        return Some(best_key.clone());
    }
    let ties = tied_at(map, best);
    doc_words
        .iter()
        .find_map(|words| first_in_ties(&build_doc_graph(words, cache, true), &ties))
}

/// Both the global candidate counts and the inverted index are sharded by
/// candidate hash so the per-step count updates can be applied in parallel
/// (shard tasks never touch the same candidate). Number of shards is a
/// power of two so routing is a mask.
const CAND_SHARDS: usize = 16;

fn cand_shard(cand: &[GraphV]) -> usize {
    use std::hash::{Hash, Hasher};
    let mut h = rustc_hash::FxHasher::default();
    cand.hash(&mut h);
    (h.finish() as usize) & (CAND_SHARDS - 1)
}

/// `best_and_tie` across shards: same single-pass logic, with cross-shard
/// score collisions folded into the tie flag.
fn best_and_tie_sharded(
    shards: &[FxHashMap<Vec<GraphV>, usize>],
) -> Option<(&Vec<GraphV>, bool, usize)> {
    let mut best: Option<(&Vec<GraphV>, bool, usize)> = None;
    for shard in shards {
        let Some((k, tie, s)) = best_and_tie(shard) else {
            continue;
        };
        match &mut best {
            None => best = Some((k, tie, s)),
            Some((bk, btie, bs)) => {
                if s > *bs {
                    *bk = k;
                    *btie = tie;
                    *bs = s;
                } else if s == *bs {
                    *btie = true;
                }
            }
        }
    }
    best
}

/// Tie-break over unique entries in first-occurrence order, using the inverted
/// index to jump straight to the earliest entry holding a tied candidate.
fn pick_best_entries(
    global: &[FxHashMap<Vec<GraphV>, usize>],
    entries: &[WordEntry],
    index: &[FxHashMap<Vec<GraphV>, FxHashSet<u32>>],
) -> Option<Vec<GraphV>> {
    let (best_key, tie, best) = best_and_tie_sharded(global)?;
    if !tie {
        return Some(best_key.clone());
    }
    let mut ties: FxHashSet<Vec<GraphV>> = FxHashSet::default();
    for shard in global {
        ties.extend(tied_at(shard, best));
    }
    let mi = ties
        .iter()
        .filter_map(|c| index[cand_shard(c)].get(c).and_then(|v| v.iter().min()).copied())
        .min()?;
    first_in_ties(&entries[mi as usize].graph, &ties)
}

fn make_token(nodes: &[GraphV]) -> GraphV {
    nodes
        .iter()
        .skip(1)
        .fold(nodes[0].clone(), |acc, n| acc.add(n))
}

fn count_merges_into(
    subs: &[GraphV],
    active: &[bool],
    global: &mut FxHashMap<Vec<GraphV>, usize>,
) {
    global.clear();

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

    for partial in chunk_maps {
        for (merge, count) in partial {
            *global.entry(merge).or_insert(0) += count;
        }
    }
}

fn apply_merge_parallel(
    subs: &mut [GraphV],
    active: &mut [bool],
    token: &GraphV,
    merge: &[GraphV],
) {
    let first_node = &merge[0];
    let changes: Vec<(usize, GraphV)> = subs
        .par_iter()
        .enumerate()
        .filter_map(|(i, sg)| {
            if !active[i] || !sg.contains_node(first_node) {
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

// Share equal non-Node children of Unconn doc Seqs behind one Arc corpus-wide
// and return the ptr -> word registry over them.
fn canonicalize_subs(subs: &mut [GraphV]) -> FxHashMap<usize, GraphV> {
    let mut canon: FxHashMap<GraphV, GraphV> = FxHashMap::default();
    for sub in subs.iter_mut() {
        if let GraphV::Seq(nodes) = sub {
            let new_nodes: Vec<GraphV> = nodes
                .iter()
                .map(|n| {
                    if matches!(n, GraphV::Node(_)) {
                        n.clone()
                    } else {
                        canon.entry(n.clone()).or_insert_with(|| n.clone()).clone()
                    }
                })
                .collect();
            *sub = GraphV::new_seq(new_nodes);
        }
    }
    canon
        .into_values()
        .filter_map(|g| crate::graph::arc_ptr(&g).map(|p| (p, g)))
        .collect()
}

// Apply one merge across all docs: merge each registered word once (parallel),
// then rewrite docs via pointer lookup. Mirrors `seq_try_merge` exactly:
// top-level matches are found against the ORIGINAL children (a child that only
// collapses to a matching Node during this same merge is not re-matched), and
// a doc collapsing to a single element via top-level matches is returned
// without descending into it. Unregistered children (or a stale registry) fall
// back to a direct try_merge, so the registry is only ever an accelerator.
fn apply_premerge(
    subs: &mut [GraphV],
    registry: &mut FxHashMap<usize, GraphV>,
    token: &GraphV,
    merge: &[GraphV],
) {
    // Only words that merge land in `changed`; still-registered pointers are
    // known unchanged, so no full result map is built or scanned per call.
    let changed: FxHashMap<usize, GraphV> = registry
        .par_iter()
        .filter_map(|(&p, g)| g.try_merge(token, merge).map(|new_g| (p, new_g)))
        .collect();
    for (p, new_g) in &changed {
        registry.remove(p);
        if let Some(np) = crate::graph::arc_ptr(new_g) {
            registry.insert(np, new_g.clone());
        }
    }
    let registry = &*registry;

    let m = merge.len();
    subs.par_iter_mut().for_each(|doc| {
        let GraphV::Seq(nodes_arc) = doc else {
            if let Some(new_doc) = doc.try_merge(token, merge) {
                *doc = new_doc;
            }
            return;
        };
        let n = nodes_arc.len();

        // Cheap scan first: top-level matches and per-child replacements.
        // Untouched docs return without cloning anything.
        let mut top_match = false;
        let mut i = 0;
        while i + m <= n {
            if nodes_arc[i..i + m] == *merge {
                top_match = true;
                break;
            }
            i += 1;
        }
        let mut replacements: Vec<(usize, GraphV)> = Vec::new();
        for (i, c) in nodes_arc.iter().enumerate() {
            if matches!(c, GraphV::Node(_)) {
                continue;
            }
            let merged = match crate::graph::arc_ptr(c) {
                Some(p) => match changed.get(&p) {
                    Some(new_g) => Some(new_g.clone()),
                    None if registry.contains_key(&p) => None,
                    None => c.try_merge(token, merge),
                },
                None => c.try_merge(token, merge),
            };
            if let Some(new_c) = merged {
                replacements.push((i, new_c));
            }
        }
        if !top_match && replacements.is_empty() {
            return;
        }

        if !top_match {
            let nodes = Arc::make_mut(nodes_arc);
            for (i, new_c) in replacements {
                nodes[i] = new_c;
            }
            return;
        }

        // Top-level match: rebuild like `seq_try_merge` — matches consume
        // the ORIGINAL children (a child that only collapses to a matching
        // Node during this same merge is not re-matched), remaining children
        // take their replacements, and a full collapse returns the single
        // element without descending into it.
        let nodes: &[GraphV] = nodes_arc;
        let replaced: FxHashMap<usize, GraphV> = replacements.into_iter().collect();
        let mut out: Vec<GraphV> = Vec::with_capacity(n);
        let mut i = 0;
        while i + m <= n {
            if nodes[i..i + m] == *merge {
                out.push(token.clone());
                i += m;
            } else {
                out.push(replaced.get(&i).cloned().unwrap_or_else(|| nodes[i].clone()));
                i += 1;
            }
        }
        while i < n {
            out.push(replaced.get(&i).cloned().unwrap_or_else(|| nodes[i].clone()));
            i += 1;
        }
        if out.len() == 1 {
            *doc = out.pop().unwrap();
        } else {
            *doc = GraphV::new_seq(out);
        }
    });
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

fn train_unconn(
    subs: &mut Vec<GraphV>,
    range_start: usize,
    num_merges: usize,
    verbose: bool,
) -> Vec<(GraphV, Vec<GraphV>)> {
    // Occurrence subgraphs repeat heavily; group identical ones into weighted
    // entries so the delta+index loop touches each unique graph once per merge.
    // Recounting is deferred so it runs once per entry, in parallel, after
    // canonicalization (entry order — the tie-break order — is fixed here).
    let mut group_of: FxHashMap<GraphV, usize> = FxHashMap::default();
    let mut entries: Vec<WordEntry> = Vec::new();
    let mut members: Vec<Vec<usize>> = Vec::new();
    for (i, sg) in subs.iter().enumerate() {
        match group_of.get(sg) {
            Some(&g) => {
                entries[g].freq += 1;
                members[g].push(i);
            }
            None => {
                let g = entries.len();
                group_of.insert(sg.clone(), g);
                entries.push(WordEntry {
                    graph: sg.clone(),
                    candidates: FxHashMap::default(),
                    freq: 1,
                    weighted: false,
                });
                members.push(vec![i]);
            }
        }
    }

    // A connected doc is one giant Seq whose children are word subgraphs, many
    // duplicated within and across docs. Share equal children by one Arc
    // corpus-wide so the per-step merge memo hits across entries (each unique
    // word merges once per step) and the weighted recount can dedup within a
    // doc; leaf Node children carry no Arc identity and stay as-is.
    let mut canon: FxHashMap<GraphV, GraphV> = FxHashMap::default();
    for entry in entries.iter_mut() {
        if let GraphV::Seq(nodes) = &entry.graph {
            let new_nodes: Vec<GraphV> = nodes
                .iter()
                .map(|n| {
                    if matches!(n, GraphV::Node(_)) {
                        n.clone()
                    } else {
                        canon.entry(n.clone()).or_insert_with(|| n.clone()).clone()
                    }
                })
                .collect();
            entry.graph = GraphV::new_seq(new_nodes);
            entry.weighted = true;
        }
    }
    let mut registry: FxHashMap<usize, GraphV> = canon
        .into_values()
        .filter_map(|g| crate::graph::arc_ptr(&g).map(|p| (p, g)))
        .collect();
    entries.par_iter_mut().for_each(|entry| entry.recount());

    let pending = train_entries_delta(
        &mut entries,
        Some(&mut registry),
        range_start,
        num_merges,
        verbose,
        true,
        &mut |token, nodes| {
            apply_merge_to_cluster_cache(token, nodes);
        },
    );

    for (entry, member_idxs) in entries.iter().zip(&members) {
        for &i in member_idxs {
            subs[i] = entry.graph.clone();
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
        let Some(nodes) = pick_best_by_scan(std::slice::from_ref(&counts), std::iter::once(&*graph)) else {
            break;
        };
        let token = make_token(&nodes);
        if verbose {
            println!("Merging {:?} count={}", nodes, counts[&nodes]);
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
    // Connected giant Seq: recount via pointer-weighted dedup of shared children.
    weighted: bool,
}

impl WordEntry {
    fn new(graph: GraphV, freq: usize) -> Self {
        let mut entry = WordEntry { graph, candidates: FxHashMap::default(), freq, weighted: false };
        entry.recount();
        entry
    }

    fn recount(&mut self) {
        self.candidates.clear();
        if self.weighted {
            if let GraphV::Seq(nodes) = &self.graph {
                crate::graph::seq_recount(nodes, &mut self.candidates);
                return;
            }
        }
        self.graph.emit_merges(&mut |m| *self.candidates.entry(m).or_insert(0) += 1);
    }

    fn apply_merge(&mut self, token: &GraphV, merge: &[GraphV]) {
        self.graph = self.graph.merge(token, merge);
        self.recount();
    }

    // try_merge variant: matches the legacy Unconn path (apply_merge_parallel),
    // where seq_try_merge no-ops minimal merges of length > 2. Returns whether
    // the graph changed; candidates are only rebuilt on change.
    fn apply_merge_try(&mut self, token: &GraphV, merge: &[GraphV]) -> bool {
        match self.graph.try_merge(token, merge) {
            Some(g) => {
                self.graph = g;
                self.recount();
                true
            }
            None => false,
        }
    }

    // Incremental apply for a top-level Seq under minimal pair merges: when the
    // pair does not match at this entry's top level, only changed children and
    // the boundary pairs adjacent to them can alter candidate counts, so the
    // decr/incr delta is produced directly (freq-scaled) and `candidates` is
    // patched in place — no full recount, no full-map diff, no map to drop.
    // Falls back (Rebuilt/None) whenever the cheap positional reasoning does
    // not apply; the fallback is byte-identical to the old path.
    fn apply_merge_try_outcome(
        &mut self,
        token: &GraphV,
        merge: &[GraphV],
        word_result: WordResult<'_>,
    ) -> ApplyOutcome {
        use crate::settings::{MAX_MERGE_SIZE, ONLY_MINIMAL_MERGES};
        use std::sync::atomic::Ordering;
        let minimal_pairs = ONLY_MINIMAL_MERGES.load(Ordering::Relaxed)
            && MAX_MERGE_SIZE.load(Ordering::Relaxed) == 2
            && merge.len() == 2;
        if minimal_pairs && matches!(&self.graph, GraphV::Seq(_)) {
            if let Some(outcome) = self.try_apply_incremental(token, merge, word_result) {
                return outcome;
            }
        }
        let old = std::mem::take(&mut self.candidates);
        if self.apply_merge_try(token, merge) {
            ApplyOutcome::Rebuilt(old)
        } else {
            self.candidates = old;
            ApplyOutcome::Unchanged
        }
    }

    // None = incremental reasoning does not apply (top-level match, or the
    // whole Seq would collapse) — caller falls back to recount+diff.
    fn try_apply_incremental(
        &mut self,
        token: &GraphV,
        merge: &[GraphV],
        word_result: WordResult<'_>,
    ) -> Option<ApplyOutcome> {
        let GraphV::Seq(nodes_arc) = &self.graph else { return None };
        let nodes: &[GraphV] = nodes_arc;
        let n = nodes.len();
        if n < 2 {
            return None;
        }
        let (m0, m1) = (&merge[0], &merge[1]);
        for i in 0..n - 1 {
            if nodes[i] == *m0 && nodes[i + 1] == *m1 {
                return None;
            }
        }

        // Walk children through an entry-local memo. Children are Arc-shared
        // corpus-wide (canonicalization in `train_unconn`), so repeated words
        // in this doc merge once and every occurrence maps to the same new
        // Arc — per-ptr weights carry over exactly as seq_recount would count
        // them. (A step-wide mutex-sharded memo was measured slower than this
        // lock-free local one.)
        let mut local_memo = crate::graph::MergeMemo::default();
        // position -> replacement, applied in place at the end (no full
        // children Vec rebuild for the common no-top-match case).
        let mut replacements: Vec<(usize, GraphV)> = Vec::new();
        // old_ptr -> (old child, new child, occurrence count) for locally
        // merged children (no precomputed step delta)
        let mut changed_ptrs: FxHashMap<usize, (GraphV, GraphV, isize)> = FxHashMap::default();
        // old_ptr -> (precomputed word delta, occurrence count)
        type SharedDelta<'a> = (&'a [(Vec<GraphV>, isize)], isize);
        let mut changed_shared: FxHashMap<usize, SharedDelta<'_>> = FxHashMap::default();
        // ptr-less children (e.g. Tree) are counted per occurrence, matching seq_recount
        let mut changed_plain: Vec<(GraphV, GraphV)> = Vec::new();

        for (i, child) in nodes.iter().enumerate() {
            if matches!(child, GraphV::Node(_)) {
                continue;
            }
            let ptr = crate::graph::arc_ptr(child);
            // Precomputed step delta available? (word came from the registry)
            let mut shared_delta: Option<&[(Vec<GraphV>, isize)]> = None;
            let merged = match (ptr, word_result) {
                (Some(p), Some((changed, registry))) => match changed.get(&p) {
                    Some((new_g, deltas)) => {
                        shared_delta = Some(deltas);
                        Some(new_g.clone())
                    }
                    // Still registered means this word does not contain the
                    // merge; a miss on both maps is a diverged (unregistered)
                    // child that must be merged locally.
                    None if registry.contains_key(&p) => None,
                    None => child.try_merge_m(token, merge, &mut local_memo),
                },
                _ => child.try_merge_m(token, merge, &mut local_memo),
            };
            if let Some(new_child) = merged {
                match (ptr, shared_delta) {
                    (Some(p), Some(deltas)) => {
                        changed_shared
                            .entry(p)
                            .and_modify(|e| e.1 += 1)
                            .or_insert((deltas, 1isize));
                    }
                    (Some(p), None) => {
                        changed_ptrs
                            .entry(p)
                            .and_modify(|e| e.2 += 1)
                            .or_insert_with(|| (child.clone(), new_child.clone(), 1));
                    }
                    (None, _) => changed_plain.push((child.clone(), new_child.clone())),
                }
                replacements.push((i, new_child));
            }
        }

        if replacements.is_empty() {
            return Some(ApplyOutcome::Unchanged);
        }

        let mut delta: FxHashMap<Vec<GraphV>, isize> = FxHashMap::default();
        for (deltas, w) in changed_shared.values() {
            for (cand, d) in *deltas {
                *delta.entry(cand.clone()).or_insert(0) += d * w;
            }
        }
        for (old_c, new_c, w) in changed_ptrs.values() {
            let w = *w;
            old_c.emit_merges(&mut |m| *delta.entry(m).or_insert(0) -= w);
            new_c.emit_merges(&mut |m| *delta.entry(m).or_insert(0) += w);
        }
        for (old_c, new_c) in &changed_plain {
            old_c.emit_merges(&mut |m| *delta.entry(m).or_insert(0) -= 1);
            new_c.emit_merges(&mut |m| *delta.entry(m).or_insert(0) += 1);
        }

        // Boundary pairs: only pair positions touching a changed child can
        // differ. seq_recount emits pair (i, i+1) iff both children are leaf
        // Nodes (minimal pair mode).
        let repl_at: FxHashMap<usize, &GraphV> =
            replacements.iter().map(|(i, g)| (*i, g)).collect();
        let new_at = |x: usize| -> &GraphV { repl_at.get(&x).copied().unwrap_or(&nodes[x]) };
        let mut starts: FxHashSet<usize> = FxHashSet::default();
        for &(c, _) in &replacements {
            if c > 0 {
                starts.insert(c - 1);
            }
            if c + 1 < n {
                starts.insert(c);
            }
        }
        for &p in &starts {
            if nodes[p].is_node() && nodes[p + 1].is_node() {
                *delta
                    .entry(vec![nodes[p].clone(), nodes[p + 1].clone()])
                    .or_insert(0) -= 1;
            }
            let (np, nq) = (new_at(p), new_at(p + 1));
            if np.is_node() && nq.is_node() {
                *delta
                    .entry(vec![np.clone(), nq.clone()])
                    .or_insert(0) += 1;
            }
        }

        let freq = self.freq;
        let mut ops: Vec<CandOp> = Vec::new();
        for (cand, d) in delta {
            match d.cmp(&0) {
                std::cmp::Ordering::Equal => {}
                std::cmp::Ordering::Less => {
                    let amt = (-d) as usize;
                    let cur = self
                        .candidates
                        .get_mut(&cand)
                        .expect("incremental delta must decrement an existing candidate");
                    *cur -= amt;
                    let removed = *cur == 0;
                    if removed {
                        self.candidates.remove(&cand);
                    }
                    ops.push((cand_shard(&cand) as u8, true, removed, amt * freq, cand));
                }
                std::cmp::Ordering::Greater => {
                    let d = d as usize;
                    let cur = self.candidates.entry(cand.clone()).or_insert(0);
                    let added = *cur == 0;
                    *cur += d;
                    ops.push((cand_shard(&cand) as u8, false, added, d * freq, cand));
                }
            }
        }

        // The scan borrows are done; write the replacements in place. Entry
        // graphs are uniquely owned after canonicalization, so make_mut does
        // not clone.
        let GraphV::Seq(nodes_arc) = &mut self.graph else { unreachable!() };
        let nodes_mut = Arc::make_mut(nodes_arc);
        for (i, new_child) in replacements {
            nodes_mut[i] = new_child;
        }
        Some(ApplyOutcome::Delta(ops))
    }
}

// A changed word's replacement graph plus its once-per-step candidate delta,
// and the (changed, registry) pair the apply path resolves children against.
type WordChange = (GraphV, Vec<(Vec<GraphV>, isize)>);
type WordResult<'a> = Option<(&'a FxHashMap<usize, WordChange>, &'a FxHashMap<usize, GraphV>)>;

// (shard, is_decr, left/entered-entry, freq-scaled amount, candidate) — one
// global/index count update, pre-routed to its candidate shard.
type CandOp = (u8, bool, bool, usize, Vec<GraphV>);

enum ApplyOutcome {
    Unchanged,
    // Old candidate map; the new one has been rebuilt into `candidates`.
    Rebuilt(FxHashMap<Vec<GraphV>, usize>),
    // Count updates ready for the sharded global/index apply.
    Delta(Vec<CandOp>),
}


fn build_word_entries(
    doc_words: &[Vec<String>],
    cache: &HashMap<String, GraphV>,
) -> Vec<WordEntry> {
    // First-occurrence order (like the reference's dict.fromkeys) so the
    // tie-break scan over entries matches traversal order.
    let mut order: Vec<&str> = Vec::new();
    let mut freq_map: FxHashMap<&str, usize> = FxHashMap::default();
    for words in doc_words {
        for w in words {
            match freq_map.get_mut(w.as_str()) {
                Some(f) => *f += 1,
                None => {
                    freq_map.insert(w.as_str(), 1);
                    order.push(w.as_str());
                }
            }
        }
    }
    order
        .into_iter()
        .filter_map(|w| cache.get(w).map(|g| WordEntry::new(g.clone(), freq_map[w])))
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

fn build_global_counts(entries: &[WordEntry]) -> Vec<FxHashMap<Vec<GraphV>, usize>> {
    let n_threads = rayon::current_num_threads().max(1);
    let chunk_size = (entries.len() / n_threads).max(1);

    let chunk_maps: Vec<Vec<FxHashMap<Vec<GraphV>, usize>>> = entries
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut local: Vec<FxHashMap<Vec<GraphV>, usize>> =
                (0..CAND_SHARDS).map(|_| FxHashMap::default()).collect();
            for entry in chunk {
                for (merge, &count) in &entry.candidates {
                    *local[cand_shard(merge)].entry(merge.clone()).or_insert(0) +=
                        count * entry.freq;
                }
            }
            local
        })
        .collect();

    let mut global: Vec<FxHashMap<Vec<GraphV>, usize>> =
        (0..CAND_SHARDS).map(|_| FxHashMap::default()).collect();
    global.par_iter_mut().enumerate().for_each(|(s, shard)| {
        for chunk in &chunk_maps {
            for (merge, &count) in &chunk[s] {
                *shard.entry(merge.clone()).or_insert(0) += count;
            }
        }
    });
    global
}

// Inverted index: candidate -> entry indices that currently contain it.
// Each entry appears at most once per candidate (candidate keys are unique per entry).
fn build_candidate_index(entries: &[WordEntry]) -> Vec<FxHashMap<Vec<GraphV>, FxHashSet<u32>>> {
    let n_threads = rayon::current_num_threads().max(1);
    let chunk_size = (entries.len() / n_threads).max(1);

    let chunk_maps: Vec<Vec<FxHashMap<Vec<GraphV>, FxHashSet<u32>>>> = entries
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            let base = (chunk_idx * chunk_size) as u32;
            let mut local: Vec<FxHashMap<Vec<GraphV>, FxHashSet<u32>>> =
                (0..CAND_SHARDS).map(|_| FxHashMap::default()).collect();
            for (offset, entry) in chunk.iter().enumerate() {
                for cand in entry.candidates.keys() {
                    local[cand_shard(cand)]
                        .entry(cand.clone())
                        .or_default()
                        .insert(base + offset as u32);
                }
            }
            local
        })
        .collect();

    let mut index: Vec<FxHashMap<Vec<GraphV>, FxHashSet<u32>>> =
        (0..CAND_SHARDS).map(|_| FxHashMap::default()).collect();
    index.par_iter_mut().enumerate().for_each(|(s, shard)| {
        for chunk in &chunk_maps {
            for (cand, idxs) in &chunk[s] {
                shard.entry(cand.clone()).or_default().extend(idxs.iter().copied());
            }
        }
    });
    index
}

// Persistent global counts + inverted index with delta updates. Per merge step
// only the entries containing the chosen candidate are touched, and `global` is
// patched by diffing each entry's old vs new candidates instead of being rebuilt.
// `on_merge` fires once per merge in order (used for cluster-cache side effects).
// `use_try` selects try_merge semantics (legacy Unconn path) over merge; with
// try_merge a no-op leaves the entry and counts untouched.
fn train_entries_delta(
    entries: &mut Vec<WordEntry>,
    mut registry: Option<&mut FxHashMap<usize, GraphV>>,
    range_start: usize,
    num_merges: usize,
    verbose: bool,
    use_try: bool,
    on_merge: &mut dyn FnMut(&GraphV, &[GraphV]),
) -> Vec<(GraphV, Vec<GraphV>)> {
    let mut pending = Vec::new();
    let mut global = build_global_counts(entries);
    let mut index = build_candidate_index(entries);

    for _ in range_start..num_merges {
        let Some(nodes) = pick_best_entries(&global, entries, &index) else {
            break;
        };
        let token = make_token(&nodes);
        if verbose {
            println!("Merging {:?} count={}", nodes, global[cand_shard(&nodes)][&nodes]);
        }

        let affected: Vec<u32> = index[cand_shard(&nodes)]
            .get(&nodes)
            .map(|set| set.iter().copied().collect())
            .unwrap_or_default();

        // Merge each registered unique word once per step; docs then resolve
        // their children by pointer lookup. Some(None) means "known unchanged"
        // (skips any local work); a miss falls back to a local try_merge, so a
        // stale registry (entries diverged through the Rebuilt path) only
        // costs time, never correctness.
        // Only words that actually merge land in `changed`; "still in the
        // registry" is the known-unchanged signal, so no per-step full map
        // rebuild or full refresh scan is paid.
        let changed_words: Option<FxHashMap<usize, WordChange>> =
            registry.as_deref().map(|reg| {
                reg.par_iter()
                    .filter_map(|(&p, g)| {
                        let new_g = g.try_merge(&token, &nodes)?;
                        // This word's candidate delta, computed once per step
                        // and shared by every doc containing the word.
                        let mut d: FxHashMap<Vec<GraphV>, isize> = FxHashMap::default();
                        g.emit_merges(&mut |m| *d.entry(m).or_insert(0) -= 1);
                        new_g.emit_merges(&mut |m| *d.entry(m).or_insert(0) += 1);
                        let deltas: Vec<(Vec<GraphV>, isize)> =
                            d.into_iter().filter(|&(_, v)| v != 0).collect();
                        Some((p, (new_g, deltas)))
                    })
                    .collect()
            });
        if let (Some(reg), Some(changed)) = (registry.as_deref_mut(), changed_words.as_ref()) {
            for (p, (new_g, _)) in changed {
                reg.remove(p);
                if let Some(np) = crate::graph::arc_ptr(new_g) {
                    reg.insert(np, new_g.clone());
                }
            }
        }
        let word_result: WordResult<'_> = match (changed_words.as_ref(), registry.as_deref()) {
            (Some(c), Some(r)) => Some((c, r)),
            _ => None,
        };

        // Apply the merge to each affected entry, producing either a ready
        // decr/incr delta (incremental path) or the old candidate map for a
        // full diff (fallback). This is the hot phase, so run it in parallel
        // across affected entries; the bookkeeping below stays serial since it
        // mutates `global` and `index`. Order does not affect output: count
        // updates are additive and pick_best_entries reads index vectors via
        // `min()`.
        let apply_one = |entry: &mut WordEntry| -> Option<ApplyOutcome> {
            let outcome = if use_try {
                entry.apply_merge_try_outcome(&token, &nodes, word_result)
            } else {
                let old = std::mem::take(&mut entry.candidates);
                entry.apply_merge(&token, &nodes);
                ApplyOutcome::Rebuilt(old)
            };
            match outcome {
                ApplyOutcome::Unchanged => None,
                other => Some(other),
            }
        };

        let changes: Vec<(u32, ApplyOutcome)> = if affected.len() >= 64 {
            let mut mask = vec![false; entries.len()];
            for &idx in &affected {
                mask[idx as usize] = true;
            }
            entries
                .par_iter_mut()
                .enumerate()
                .filter_map(|(i, entry)| {
                    if !mask[i] {
                        return None;
                    }
                    apply_one(entry).map(|outcome| (i as u32, outcome))
                })
                .collect()
        } else {
            affected
                .iter()
                .filter_map(|&idx| apply_one(&mut entries[idx as usize]).map(|o| (idx, o)))
                .collect()
        };

        // Rebuilt entries still need the full old-vs-new map diff. Only a
        // handful of candidates actually change, but the diff deep-hashes both
        // full maps, so it runs in parallel; `into_par_iter` also lets each
        // worker free its entry's old map (dropping those maps costs more than
        // the diff itself when done serially at end of scope). Delta outcomes
        // pass straight through.
        let diff_one = |idx: u32, old: &FxHashMap<Vec<GraphV>, usize>| -> Vec<CandOp> {
            let entry = &entries[idx as usize];
            let freq = entry.freq;
            let new = &entry.candidates;
            let mut ops = Vec::new();
            for (cand, &old_c) in old {
                let new_c = new.get(cand).copied().unwrap_or(0);
                if new_c < old_c {
                    ops.push((
                        cand_shard(cand) as u8,
                        true,
                        new_c == 0,
                        (old_c - new_c) * freq,
                        cand.clone(),
                    ));
                }
            }
            for (cand, &new_c) in new {
                let old_c = old.get(cand).copied().unwrap_or(0);
                if new_c > old_c {
                    ops.push((
                        cand_shard(cand) as u8,
                        false,
                        old_c == 0,
                        (new_c - old_c) * freq,
                        cand.clone(),
                    ));
                }
            }
            ops
        };
        let resolve = |(idx, outcome): (u32, ApplyOutcome)| -> (u32, Vec<CandOp>) {
            match outcome {
                ApplyOutcome::Rebuilt(old) => (idx, diff_one(idx, &old)),
                ApplyOutcome::Delta(ops) => (idx, ops),
                ApplyOutcome::Unchanged => unreachable!("filtered out in apply phase"),
            }
        };

        let diffs: Vec<(u32, Vec<CandOp>)> = if changes.len() >= 64 {
            changes.into_par_iter().map(resolve).collect()
        } else {
            changes.into_iter().map(resolve).collect()
        };

        // Route ops to their candidate shard (tag precomputed in parallel
        // above), then apply shards concurrently. Any op order yields the same
        // final counts: updates are additive, an entry never emits the same
        // candidate twice in one step, and a count only reaches zero on its
        // final decrement.
        let mut buckets: Vec<Vec<(u32, CandOp)>> =
            (0..CAND_SHARDS).map(|_| Vec::new()).collect();
        for (idx, ops) in diffs {
            for op in ops {
                buckets[op.0 as usize].push((idx, op));
            }
        }
        buckets
            .into_par_iter()
            .zip(global.par_iter_mut().zip(index.par_iter_mut()))
            .for_each(|(bucket, (gshard, ishard))| {
                for (idx, (_, is_decr, flag, amount, cand)) in bucket {
                    if is_decr {
                        let g = gshard.get_mut(&cand).unwrap();
                        *g -= amount;
                        if *g == 0 {
                            gshard.remove(&cand);
                        }
                        if flag {
                            if let Some(v) = ishard.get_mut(&cand) {
                                v.remove(&idx);
                            }
                        }
                    } else {
                        if flag {
                            ishard.entry(cand.clone()).or_default().insert(idx);
                        }
                        *gshard.entry(cand).or_insert(0) += amount;
                    }
                }
            });
        on_merge(&token, &nodes);
        pending.push((token, nodes));
    }
    pending
}

// Disconnected streaming: word-level candidate caching via the delta+index loop.
fn train_streaming_disconnected(
    entries: &mut Vec<WordEntry>,
    range_start: usize,
    num_merges: usize,
    verbose: bool,
) -> Vec<(GraphV, Vec<GraphV>)> {
    train_entries_delta(entries, None, range_start, num_merges, verbose, false, &mut |_, _| {})
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

        let Some(nodes) = pick_best_docs(&global, doc_words, cache) else {
            break;
        };
        let token = make_token(&nodes);
        if verbose {
            println!("Merging {:?} count={}", nodes, global[&nodes]);
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

        let node_count = |entries: &[WordEntry]| -> usize {
            entries.par_iter().map(|e| e.graph.node_count() * e.freq).sum()
        };

        let mut pending = Vec::new();
        let mut xs = vec![0usize];
        let mut ys = vec![node_count(&entries)];

        for i in range_start..num_merges {
            let global = build_global_counts(&entries);
            let Some(nodes) = pick_best_by_scan(&global, entries.iter().map(|e| &e.graph)) else { break };
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

        let Some(nodes) = pick_best_docs(&global, doc_words, &cache) else { break };
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
    // Arc-ptr -> canonical word graph for Unconn children, built lazily on the
    // first apply_merge so repeated pre-merge application (SuperBPE phase 1 ->
    // phase 2) merges each unique word once per call instead of once per doc.
    // Purely an accelerator: unknown pointers fall back to a direct try_merge.
    premerge_registry: Option<FxHashMap<usize, GraphV>>,
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
                premerge_registry: None,
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
                    premerge_registry: None,
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
        // Training rewrites the graph with its own canonical Arcs; drop the
        // pre-merge accelerator so it stops keeping the old ones alive.
        self.premerge_registry = None;
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
                let mut counts: FxHashMap<Vec<GraphV>, usize> = FxHashMap::default();

                for i in range_start..num_merges {
                    count_merges_into(subs, &active, &mut counts);
                    let Some(nodes) = pick_best_by_scan(
                        std::slice::from_ref(&counts),
                        subs.iter().zip(active.iter()).filter(|(_, &a)| a).map(|(g, _)| g),
                    ) else {
                        break;
                    };
                    let token = make_token(&nodes);
                    apply_merge_parallel(subs, &mut active, &token, &nodes);
                    apply_merge_to_cluster_cache(&token, &nodes);
                    pending.push((token, nodes));

                    let merge_num = i + 1;
                    if merge_num % sample_every == 0 || merge_num == num_merges {
                        let nc: usize = subs.iter().map(|sg| sg.node_count()).sum();
                        xs.push(merge_num);
                        ys.push(nc);
                    }
                }
            } else {
                let mut counts: FxHashMap<Vec<GraphV>, usize> = FxHashMap::default();
                for i in range_start..num_merges {
                    counts.clear();
                    graph.emit_merges(&mut |m| *counts.entry(m).or_insert(0) += 1);
                    let Some(nodes) = pick_best_by_scan(std::slice::from_ref(&counts), std::iter::once(&*graph)) else {
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
            premerge_registry: None,
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
        self.premerge_registry = None;
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
        // Applying pre-merges to a corpus (e.g. SuperBPE phase-1 → phase-2)
        // walks every document. Canonicalize children once, then merge each
        // unique word once per call and rebuild docs by pointer lookup.
        if let GraphV::Unconn(subs_arc) = &mut self.graph {
            let subs = Arc::make_mut(subs_arc);
            let registry = self
                .premerge_registry
                .get_or_insert_with(|| canonicalize_subs(subs));
            apply_premerge(subs, registry, &token_g, &merge_nodes);
        } else {
            self.graph = self.graph.merge(&token_g, &merge_nodes);
        }
        self.merges.push((token_g, merge_nodes));
        Ok(())
    }
}
