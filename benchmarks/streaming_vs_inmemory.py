"""Compare in-memory vs streaming vs word-cached merge counting.

In-memory:   build all graphs once, keep full doc graphs, count from big graph.
Streaming:   rebuild each doc graph from scratch each step, dispose after counting.
Word-cached: cache word→graph, apply only new merge per step, stream docs but hit cache.
"""

import time
import tracemalloc
from collections import Counter
from functools import reduce
from itertools import islice

from datasets import load_dataset

from complex_tokenization.graphs.settings import GraphSettings
from complex_tokenization.graphs.units import register_script, utf8_clusters
from complex_tokenization.graphs.words import GPTPretokenizer, pretokenize, words
from complex_tokenization.languages.chinese.graph import chinese_character_to_graph
from complex_tokenization.trainer import Trainer

register_script("Han", chinese_character_to_graph)

NUM_MERGES = 5
NUM_DOCS = 10


def load_texts(n):
    ds = load_dataset("wikimedia/wikipedia", "20231101.zh", split="train", streaming=True)
    return [row["text"][:500] for row in islice(ds, n) if row["text"]]


def fmt_mem(nbytes):
    if nbytes > 1024 * 1024:
        return f"{nbytes / 1024 / 1024:.1f} MB"
    return f"{nbytes / 1024:.1f} KB"


def train_inmemory(texts, num_merges):
    GraphSettings.ONLY_MINIMAL_MERGES = True
    GraphSettings.MAX_MERGE_SIZE = 2

    tracemalloc.start()
    t0 = time.time()

    graphs = tuple(words(t, connected=False, pretokenizer=GPTPretokenizer) for t in texts)
    trainer = Trainer(graphs=graphs)

    merges = []
    for _step in range(num_merges):
        counts = Counter(trainer.graph.get_merges())
        if not counts:
            break
        best = max(counts, key=lambda k: (len(k) - 1) * counts[k])
        token = reduce(lambda x, y: x + y, best)
        trainer.graph = trainer.graph.merge(token, best)
        merges.append((token, best))

    total_time = time.time() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return merges, total_time, peak_mem


def train_streaming(texts, num_merges):
    GraphSettings.ONLY_MINIMAL_MERGES = True
    GraphSettings.MAX_MERGE_SIZE = 2

    tracemalloc.start()
    t0 = time.time()

    merges = []
    for _step in range(num_merges):
        counts = Counter()
        for text in texts:
            graph = words(text, connected=False, pretokenizer=GPTPretokenizer)
            for token, nodes in merges:
                graph = graph.merge(token, nodes)
            counts.update(graph.get_merges())

        if not counts:
            break
        best = max(counts, key=lambda k: (len(k) - 1) * counts[k])
        token = reduce(lambda x, y: x + y, best)
        merges.append((token, best))

    total_time = time.time() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return merges, total_time, peak_mem


def train_word_cached(texts, num_merges):
    GraphSettings.ONLY_MINIMAL_MERGES = True
    GraphSettings.MAX_MERGE_SIZE = 2

    tracemalloc.start()
    t0 = time.time()

    cache = {}
    word_freq = Counter()
    merges = []

    for _step in range(num_merges):
        counts = Counter()

        for text in texts:
            for w in pretokenize(text, GPTPretokenizer):
                word_freq[w] += 1
                if w not in cache:
                    g = utf8_clusters(w)
                    for token, nodes in merges:
                        g = g.merge(token, nodes)
                    cache[w] = g
                for m in cache[w].get_merges():
                    counts[m] += 1

        if not counts:
            break
        best = max(counts, key=lambda k: (len(k) - 1) * counts[k])
        token = reduce(lambda x, y: x + y, best)
        merges.append((token, best))

        for w in cache:
            cache[w] = cache[w].merge(token, best)

    total_time = time.time() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return merges, total_time, peak_mem


def main():
    print(f"Loading {NUM_DOCS} docs...")
    texts = load_texts(NUM_DOCS)
    total_chars = sum(len(t) for t in texts)
    print(f"  {total_chars:,} chars total\n")

    print(f"--- In-memory ({NUM_MERGES} merges) ---")
    im_merges, im_time, im_mem = train_inmemory(texts, NUM_MERGES)
    print(f"  Time:   {im_time:.2f}s")
    print(f"  Memory: {fmt_mem(im_mem)}")

    print(f"\n--- Streaming ({NUM_MERGES} merges) ---")
    st_merges, st_time, st_mem = train_streaming(texts, NUM_MERGES)
    print(f"  Time:   {st_time:.2f}s")
    print(f"  Memory: {fmt_mem(st_mem)}")

    print(f"\n--- Word-cached ({NUM_MERGES} merges) ---")
    wc_merges, wc_time, wc_mem = train_word_cached(texts, NUM_MERGES)
    print(f"  Time:   {wc_time:.2f}s")
    print(f"  Memory: {fmt_mem(wc_mem)}")

    im_bytes = [tuple(bytes(n) for n in ns) for _, ns in im_merges]
    st_bytes = [tuple(bytes(n) for n in ns) for _, ns in st_merges]
    wc_bytes = [tuple(bytes(n) for n in ns) for _, ns in wc_merges]

    print("\n--- Summary ---")
    print(f"  {'':25s} {'Time':>8s} {'Memory':>10s}")
    print(f"  {'In-memory':25s} {im_time:>7.2f}s {fmt_mem(im_mem):>10s}")
    print(f"  {'Streaming':25s} {st_time:>7.2f}s {fmt_mem(st_mem):>10s}")
    print(f"  {'Word-cached':25s} {wc_time:>7.2f}s {fmt_mem(wc_mem):>10s}")
    print(f"\n  Streaming identical:  {im_bytes == st_bytes}")
    print(f"  Word-cached identical: {im_bytes == wc_bytes}")


if __name__ == "__main__":
    main()
