"""Test that singletons produce identical results and measure their overhead."""

import time

from complex_tokenization.graph import GraphVertex
from complex_tokenization.graphs.settings import GraphSettings
from complex_tokenization.graphs.units import utf8_clusters
from complex_tokenization.graphs.words import words
from complex_tokenization.trainer import Trainer


def train_with_settings(texts, use_singletons, num_merges=20):
    GraphVertex._instances.clear()
    GraphSettings.USE_SINGLETONS = use_singletons
    GraphSettings.ONLY_MINIMAL_MERGES = True
    GraphSettings.MAX_MERGE_SIZE = 2

    graphs = tuple([words(text, connected=False, units=utf8_clusters) for text in texts])
    trainer = Trainer(graphs=graphs)
    trainer.train(num_merges=num_merges)
    return trainer.get_merges()


class TestSingletonPerformance:
    def test_singletons_produce_same_merges(self):
        texts = ["the teacher teaches the thick thing", "hello world test"]
        merges_off = train_with_settings(texts, use_singletons=False)
        merges_on = train_with_settings(texts, use_singletons=True)
        assert merges_off == merges_on

    def test_singleton_new_is_slower_than_regular_new(self):
        """The cache key construction in __new__ is more expensive than just
        creating a fresh frozen dataclass. This is the root cause of singleton
        overhead: every Node() call pays for building a tuple key and a dict
        lookup, which costs more than allocating a small frozen object."""
        from complex_tokenization.graph import Node

        n = 50_000
        GraphSettings.USE_SINGLETONS = False
        start = time.perf_counter()
        for i in range(n):
            Node(value=bytes([i % 256]))
        time_off = time.perf_counter() - start

        GraphVertex._instances.clear()
        GraphSettings.USE_SINGLETONS = True
        start = time.perf_counter()
        for i in range(n):
            Node(value=bytes([i % 256]))
        time_on = time.perf_counter() - start

        assert time_on > time_off, (
            f"Expected singleton __new__ to be slower: {time_on:.4f}s vs {time_off:.4f}s"
        )
