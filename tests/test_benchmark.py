"""Benchmark tests to detect performance regressions."""

import time

import pytest

from complex_tokenization.tokenizer import BNETokenizer, BoundlessBPETokenizer, BPETokenizer, SuperBPETokenizer
from tests.utils import text_dataset, train_huggingface_tokenizer


@pytest.fixture(scope="module")
def small_dataset():
    return list(text_dataset(max_samples=10))


class TestBenchmarkSmall:
    def test_bpe_matches_huggingface_merges(self, small_dataset):
        ours = BPETokenizer().train(small_dataset, num_merges=10)
        hf = train_huggingface_tokenizer(small_dataset, num_merges=10)
        hf_normalized = [(m[0].replace("Ġ", " "), m[1]) for m in hf]
        assert ours == hf_normalized

    def test_bpe_faster_than_60s(self, small_dataset):
        start = time.perf_counter()
        BPETokenizer().train(small_dataset, num_merges=50)
        elapsed = time.perf_counter() - start
        assert elapsed < 60, f"BPE training took {elapsed:.1f}s (limit: 60s)"

    def test_boundless_bpe_faster_than_60s(self, small_dataset):
        start = time.perf_counter()
        BoundlessBPETokenizer().train(small_dataset, num_merges=50)
        elapsed = time.perf_counter() - start
        assert elapsed < 60, f"Boundless BPE training took {elapsed:.1f}s (limit: 60s)"

    def test_super_bpe_faster_than_60s(self, small_dataset):
        start = time.perf_counter()
        SuperBPETokenizer().train(small_dataset, num_merges=50)
        elapsed = time.perf_counter() - start
        assert elapsed < 60, f"Super BPE training took {elapsed:.1f}s (limit: 60s)"

    def test_bne_faster_than_60s(self, small_dataset):
        start = time.perf_counter()
        BNETokenizer(n=4).train(small_dataset, num_merges=50)
        elapsed = time.perf_counter() - start
        assert elapsed < 60, f"BNE training took {elapsed:.1f}s (limit: 60s)"

    def test_all_tokenizers_produce_merges(self, small_dataset):
        num = 10
        bpe = BPETokenizer().train(small_dataset, num_merges=num)
        bne = BNETokenizer(n=4).train(small_dataset, num_merges=num)
        boundless = BoundlessBPETokenizer().train(small_dataset, num_merges=num)
        super_bpe = SuperBPETokenizer().train(small_dataset, num_merges=num)

        assert len(bpe) == num
        assert len(bne) == num
        assert len(boundless) == num
        assert len(super_bpe) == num

        for merge in bpe:
            assert len(merge) == 2
        for merge in bne:
            assert 2 <= len(merge) <= 4
