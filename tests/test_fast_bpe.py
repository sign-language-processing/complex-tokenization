"""Test FastBPETrainer produces identical results to regular BPE."""

import time

from complex_tokenization.fast_bpe_trainer import FastBPETrainer
from complex_tokenization.tokenizer import BPETokenizer


class TestFastBPECorrectness:
    def test_matches_regular_bpe_small(self):
        texts = ["the teacher teaches the thick thing"]
        fast = FastBPETrainer(texts)
        fast.train(num_merges=5)

        regular = BPETokenizer().train(texts, num_merges=5)
        assert fast.get_merges() == regular

    def test_matches_regular_bpe_medium(self):
        texts = ["the teacher teaches the thick thing " * 20] * 10
        fast = FastBPETrainer(texts)
        fast.train(num_merges=20)

        regular = BPETokenizer().train(texts, num_merges=20)
        assert fast.get_merges() == regular

    def test_empty_text(self):
        fast = FastBPETrainer([""])
        fast.train(num_merges=10)
        assert fast.get_merges() == []

    def test_single_char(self):
        fast = FastBPETrainer(["a"])
        fast.train(num_merges=10)
        assert fast.get_merges() == []


class TestFastBPEPerformance:
    def test_faster_than_regular(self):
        texts = ["the teacher teaches the thick thing " * 50] * 20
        num_merges = 100

        start = time.perf_counter()
        regular = BPETokenizer().train(texts, num_merges=num_merges)
        regular_time = time.perf_counter() - start

        start = time.perf_counter()
        fast = FastBPETrainer(texts)
        fast.train(num_merges=num_merges)
        fast_time = time.perf_counter() - start

        assert fast.get_merges() == regular
        assert fast_time < regular_time, (
            f"FastBPE ({fast_time:.3f}s) should be faster than regular ({regular_time:.3f}s)"
        )
