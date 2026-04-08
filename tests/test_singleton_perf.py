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

    def test_singletons_not_faster_than_regular(self):
        """Document that singletons are currently slower due to cache overhead."""
        texts = ["the teacher teaches the thick thing " * 10] * 5

        start = time.perf_counter()
        train_with_settings(texts, use_singletons=False, num_merges=30)
        time_off = time.perf_counter() - start

        start = time.perf_counter()
        train_with_settings(texts, use_singletons=True, num_merges=30)
        time_on = time.perf_counter() - start

        # Singletons should not be more than 3x slower (regression guard)
        assert time_on < time_off * 3, (
            f"Singletons too slow: {time_on:.3f}s vs {time_off:.3f}s"
        )
