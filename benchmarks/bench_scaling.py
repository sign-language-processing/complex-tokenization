"""Benchmark how training scales with text size and merge count."""

import time

from complex_tokenization.fast_bpe_trainer import FastBPETrainer
from complex_tokenization.graphs.settings import GraphSettings
from complex_tokenization.graphs.units import utf8_clusters
from complex_tokenization.graphs.words import words
from complex_tokenization.trainer import Trainer


def train_graph_bpe(texts, num_merges):
    GraphSettings.ONLY_MINIMAL_MERGES = True
    GraphSettings.MAX_MERGE_SIZE = 2
    GraphSettings.USE_SINGLETONS = False
    graphs = tuple(words(t, connected=False, units=utf8_clusters) for t in texts)
    trainer = Trainer(graphs=graphs)
    trainer.train(num_merges=num_merges)
    return trainer.get_merges()


def train_fast_bpe(texts, num_merges):
    fast = FastBPETrainer(texts)
    fast.train(num_merges=num_merges)
    return fast.get_merges()


BASE_TEXT = "the teacher teaches the thick thing about the theorem "


def run():
    print(f"\n{'='*80}")
    print("Scaling Benchmark: Graph BPE vs Fast BPE")
    print(f"{'='*80}")
    print(f"{'Config':30s}  {'Graph BPE':>10s}  {'Fast BPE':>10s}  {'Speedup':>8s}")
    print("-" * 80)

    for num_texts in [10, 50, 100]:
        for repeat in [10, 50]:
            for num_merges in [50, 100, 200]:
                texts = [BASE_TEXT * repeat] * num_texts
                total_chars = sum(len(t) for t in texts)

                start = time.perf_counter()
                graph_merges = train_graph_bpe(texts, num_merges)
                graph_time = time.perf_counter() - start

                start = time.perf_counter()
                fast_merges = train_fast_bpe(texts, num_merges)
                fast_time = time.perf_counter() - start

                speedup = graph_time / fast_time if fast_time > 0 else float('inf')
                match = "ok" if graph_merges == fast_merges else "MISMATCH"
                label = f"{num_texts}x{repeat}rep m={num_merges} ({total_chars:,}ch)"
                print(f"{label:30s}  {graph_time:>9.3f}s  {fast_time:>9.3f}s  {speedup:>7.1f}x  {match}")


if __name__ == "__main__":
    run()
