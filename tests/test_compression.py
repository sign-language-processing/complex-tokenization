"""Test compression ratio tracking in FastBPETrainer."""

from complex_tokenization.fast_bpe_trainer import FastBPETrainer


class TestCompressionTracking:
    def test_stats_populated(self):
        fast = FastBPETrainer(["hello world hello world"])
        fast.train(num_merges=3)
        assert len(fast.stats) == 3

    def test_compression_increases(self):
        fast = FastBPETrainer(["the teacher teaches the thick thing"] * 5)
        fast.train(num_merges=5)
        compressions = [s["compression"] for s in fast.stats]
        assert all(c >= 0 for c in compressions)
        assert compressions[-1] > compressions[0]

    def test_total_tokens_decrease(self):
        fast = FastBPETrainer(["abababababababab"] * 3)
        fast.train(num_merges=3)
        tokens = [s["total_tokens"] for s in fast.stats]
        for i in range(1, len(tokens)):
            assert tokens[i] <= tokens[i - 1]

    def test_frequency_reported(self):
        fast = FastBPETrainer(["aabbaabb"])
        fast.train(num_merges=1)
        assert fast.stats[0]["frequency"] >= 2

    def test_empty_text_no_stats(self):
        fast = FastBPETrainer([""])
        fast.train(num_merges=5)
        assert fast.stats == []
