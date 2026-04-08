"""Benchmark singleton vs non-singleton graph construction and training."""

import time
import tracemalloc

from complex_tokenization.examples.utils import text_dataset
from complex_tokenization.graph import GraphVertex
from complex_tokenization.graphs.settings import GraphSettings
from complex_tokenization.graphs.units import utf8_clusters
from complex_tokenization.graphs.words import words
from complex_tokenization.trainer import Trainer


def bench_singletons(texts, num_merges, use_singletons):
    GraphVertex._instances.clear()
    GraphSettings.USE_SINGLETONS = use_singletons
    GraphSettings.ONLY_MINIMAL_MERGES = True
    GraphSettings.MAX_MERGE_SIZE = 2

    tracemalloc.start()
    start = time.perf_counter()

    graphs = tuple([words(text, connected=False, units=utf8_clusters) for text in texts])
    graph_time = time.perf_counter() - start

    trainer = Trainer(graphs=graphs)
    trainer.train(num_merges=num_merges)

    elapsed = time.perf_counter() - start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    cache_size = len(GraphVertex._instances)
    GraphVertex._instances.clear()
    GraphSettings.USE_SINGLETONS = False

    return len(trainer.merges), elapsed, graph_time, peak_mem, cache_size


def run():
    for num_samples in [10, 50]:
        texts = list(text_dataset(max_samples=num_samples))
        total_chars = sum(len(t) for t in texts)

        print(f"\n{'='*75}")
        print(f"Singleton Benchmark: {num_samples} samples, {total_chars:,} chars")
        print(f"{'='*75}")
        print(f"{'Mode':25s}  {'Merges':>6s}  {'Graph':>7s}  {'Total':>7s}  {'Peak Mem':>10s}  {'Cache':>7s}")
        print("-" * 75)

        for num_merges in [50, 100]:
            for use_singletons in [False, True]:
                label = f"{'singleton' if use_singletons else 'no-singleton'} m={num_merges}"
                merges, elapsed, graph_time, peak_mem, cache_size = bench_singletons(
                    texts, num_merges, use_singletons
                )
                print(f"{label:25s}  {merges:>6d}  {graph_time:>6.3f}s  {elapsed:>6.3f}s  "
                      f"{peak_mem / 1024 / 1024:>8.2f} MB  {cache_size:>7d}")


if __name__ == "__main__":
    run()
