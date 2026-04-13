# complex-tokenization-fast

Rust (PyO3) drop-in replacement for `complex-tokenization`. Same API, same test suite, same merge results — 50-80x faster than the reference Python.

## Install

```bash
cd fast
pip install -e .          # uses maturin to compile Rust + install Python wrappers
```

Requires Rust toolchain and [maturin](https://www.maturin.rs/).

The reference Python package and the fast package both provide `complex_tokenization`. Install one at a time — whichever is installed last wins.

## Performance

Benchmarked on Chinese Wikipedia with IDS character decomposition (500 chars/doc):

| Config | Reference Python | **Fast (Rust)** | HuggingFace BPE |
|--------|:---:|:---:|:---:|
| 100 docs, 10 merges | 9.0s | **0.18s** (50x) | 0.03s |
| 100 docs, 50 merges | 30.0s | **0.52s** (57x) | 0.02s |
| 1000 docs, 10 merges | 90s | **1.1s** (79x) | 0.19s |
| 1000 docs, 50 merges | 300s | **3.7s** (80x) | 0.18s |

HuggingFace comparison is informational — it's flat BPE without graph structures (no tree decomposition, no FullyConnectedGraph). The 6-20x gap is the cost of the graph-based approach vs flat arrays with incremental pair counting.

## What's in Rust

All performance-critical code is Rust, exposed to Python via PyO3:

- **Graph types** (`src/graph.rs`): `Node`, `NodesSequence`, `Tree`, `FullyConnectedGraph`, `UnconnectedGraphs` — all with `merge()`, `try_merge()`, `get_merges()`, `count_merges()`, equality, hashing
- **Trainer** (`src/trainer.rs`): the merge training loop — counting, finding best, applying merges
- **Units** (`src/units.rs`): `utf8()`, `utf8_clusters()`, `characters()`, `register_script()` with grapheme cluster cache
- **Settings** (`src/settings.rs`): `GraphSettings` backed by atomics, synced from Python

## What's still Python

- Pretokenization (HuggingFace `tokenizers` library)
- `Tokenizer` / `BPETokenizer` / etc. orchestration classes
- Language-specific decomposition (Hebrew diacritics, Chinese IDS)
- Graph building (`words.py` calling `utf8_clusters` per word)

## What made it fast

In rough order of impact:

1. **Rust port of graph operations** (~15x). The core `merge()`, `get_merges()`, and `Counter()` loop moved from Python with millions of dynamic-dispatch calls to native Rust.

2. **Rayon parallelism** (~2x). `get_merges()` and `merge()` on `UnconnectedGraphs` run subgraphs in parallel across all cores. GIL released for the entire training loop.

3. **Fused `count_merges()`** (~1.3x). Instead of `get_merges()` → intermediate Vec of 300K candidates → `HashMap::count()`, each subgraph counts directly into a thread-local `FxHashMap`. No intermediate allocation.

4. **`try_merge()` with zero-alloc early return** (~1.3x). Returns `None` for unchanged subgraphs/nodes instead of allocating a new identical graph and comparing. Propagates through `NodesSequence` and `Tree` recursion.

5. **`FxHashMap`** (~1.1x). Faster hash function for merge candidate keys (small byte vectors).

6. **Flattened `UnconnectedGraphs`**. Nested `UnconnectedGraphs(UnconnectedGraphs(...))` collapsed to single level for better parallelism (6000 work items instead of 100).

7. **Grapheme cluster cache** in `utf8_clusters`. Repeated characters (e.g., "的" appearing 100 times) build the graph once, then clone via `Arc`.

## What didn't work

Approaches that were tried and reverted because they made things slower:

### Incremental counting (subgraph-level)

**Idea**: maintain a running global count `HashMap`. When a merge changes subgraph `i`, subtract its old merge counts, recompute, add the new ones. Skip unchanged subgraphs.

**Why it failed**: for Chinese with IDS decomposition, the first merges are common UTF-8 byte pairs (e.g., `0xe2 0xbf` — present in every IDS descriptor). These affect ~95% of subgraphs. The subtract/add loop does 2x the HashMap operations of a clean rebuild, for a net slowdown. Later merges are more specific (~40% affected), but the overhead doesn't recover. **Result: 1.5-2x slower**.

### Word-level deduplication

**Idea**: 5987 word graphs have only 4271 unique values. Deduplicate, process unique graphs with weights, multiply counts.

**Why it failed**: deduplicating requires hashing/comparing full graph trees to detect duplicates. Whether using `GraphV` equality (recursive deep comparison), `to_bytes()` (recursive allocation), or `Hash` (recursive traversal) — the dedup cost exceeded the 29% savings from fewer traversals. **Result: 2-3x slower**.

### Arena-based mutable graph

**Idea**: flatten the immutable recursive `GraphV` tree into a mutable arena with stable node IDs, inverted index from merge pairs to containing nodes, and O(affected) updates.

**Why it failed**: the arena nodes needed to convert back to `GraphV` for merge candidate comparison (`node_to_graphv()`), which recursively allocates at every level. This was more expensive than just traversing the original immutable graph. **Result: 13x slower**.

A correct arena implementation would need to never convert to `GraphV` during training — all comparisons should use byte content or node IDs directly. This requires changing the merge candidate key type from `Vec<GraphV>` to `Vec<Vec<u8>>` and is a deeper redesign.

### Byte-key counting

**Idea**: convert each merge candidate to its byte representation for cheaper HashMap keys (bytes are flat, no recursive hashing).

**Why it failed**: `to_bytes()` on each merge candidate traverses the graph tree and allocates a new Vec. With 300K candidates per iteration, the allocation cost exceeded the hashing savings. **Result: 3x slower**.

## The remaining gap to HuggingFace

HuggingFace's Rust BPE trainer is 6-20x faster because of a fundamentally different algorithm:

| | Our approach | HuggingFace |
|---|---|---|
| Data structure | Recursive graph trees (`NodesSequence`, `Tree`, `FullyConnectedGraph`) | Flat `Vec<u32>` per word (token ID sequences) |
| Per-merge counting | O(all nodes) — traverse every graph every iteration | O(affected pairs) — incremental update via inverted index |
| Per-merge cost | ~10ms (100 docs) | ~0.3ms (100 docs) |
| Supports trees? | Yes (Chinese IDS, Hebrew diacritics) | No (flat sequences only) |

The graph structure is essential for complex scripts (tree decomposition for Chinese characters, fully-connected graphs for Hebrew diacritics). The flat-array optimization isn't directly applicable.

The path to closing the gap: an arena-based graph where nodes are mutable, merge candidates are represented as byte tuples (not `GraphV` values), and an inverted index maps byte-tuple keys to arena node IDs. All training operations work on bytes/IDs without ever reconstructing the immutable graph representation.

## Running tests

```bash
# Install the fast version
cd fast && pip install -e .

# Run tests from outside the project root (to avoid the local complex_tokenization/ shadowing)
cd /tmp && python -m pytest /path/to/complex-tokenization/tests/ -q
```

Both the reference and fast packages pass the same 127 tests.
