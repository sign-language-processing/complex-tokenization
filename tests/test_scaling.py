"""Test that FastBPE scales well with larger inputs."""

import time

from complex_tokenization.fast_bpe_trainer import FastBPETrainer

BASE = "the teacher teaches the thick thing about the theorem "


class TestScaling:
    def test_100k_chars_under_5s(self):
        texts = [BASE * 50] * 100  # ~270k chars
        start = time.perf_counter()
        fast = FastBPETrainer(texts)
        fast.train(num_merges=25)
        elapsed = time.perf_counter() - start
        assert elapsed < 5, f"FastBPE on 270k chars took {elapsed:.1f}s (limit: 5s)"
        assert len(fast.merges) == 25

    def test_merges_scale_with_data(self):
        small = [BASE * 10] * 10
        large = [BASE * 50] * 50
        f_small = FastBPETrainer(small)
        f_small.train(num_merges=50)
        f_large = FastBPETrainer(large)
        f_large.train(num_merges=50)
        assert f_small.get_merges() == f_large.get_merges()
