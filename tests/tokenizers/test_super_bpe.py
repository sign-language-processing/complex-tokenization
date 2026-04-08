from complex_tokenization.examples.bpe import train_bpe_tokenizer
from complex_tokenization.examples.super_bpe import train_super_bpe_tokenizer
from complex_tokenization.examples.utils import text_dataset


class TestSuperBPE:
    def test_basic_super_bpe(self):
        texts = ["the teacher teaches the thick thing"]
        merges = train_super_bpe_tokenizer(texts, num_merges=4, disconnected_merges=2)
        assert len(merges) == 4

    def test_super_bpe_phase1_matches_bpe(self):
        """Phase 1 of Super BPE should produce same merges as regular BPE."""
        texts = list(text_dataset(max_samples=10))
        bpe_merges = train_bpe_tokenizer(texts, num_merges=5)
        super_merges = train_super_bpe_tokenizer(texts, num_merges=10, disconnected_merges=5)
        assert super_merges[:5] == bpe_merges

    def test_super_bpe_default_split(self):
        """Default disconnected_merges should be num_merges // 2."""
        texts = ["the teacher teaches the thick thing"]
        merges = train_super_bpe_tokenizer(texts, num_merges=4)
        assert len(merges) == 4

    def test_large_super_bpe(self):
        texts = list(text_dataset(max_samples=10))
        merges = train_super_bpe_tokenizer(texts, num_merges=10, disconnected_merges=5)
        assert len(merges) == 10
        for merge in merges:
            assert len(merge) == 2
