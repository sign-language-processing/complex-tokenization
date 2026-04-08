from complex_tokenization.examples.boundless_bpe import train_boundless_bpe_tokenizer
from complex_tokenization.examples.bpe import train_bpe_tokenizer
from complex_tokenization.examples.super_bpe import train_super_bpe_tokenizer


class TestSuperBPE:
    def test_basic_super_bpe(self):
        texts = ["the teacher teaches the thick thing"]
        merges = train_super_bpe_tokenizer(texts, num_merges=4, disconnected_merges=2)
        assert len(merges) == 4

    def test_phase1_matches_bpe(self):
        """Phase 1 of Super BPE should produce same merges as regular BPE."""
        texts = ["ab cd ab cd ab cd"] * 3
        bpe_merges = train_bpe_tokenizer(texts, num_merges=3)
        super_merges = train_super_bpe_tokenizer(texts, num_merges=5, disconnected_merges=3)
        assert super_merges[:3] == bpe_merges

    def test_extends_bpe_like_boundless(self):
        """After intra-word phase, super BPE merges across words like boundless."""
        texts = ["ab cd ab cd ab cd"]
        super_merges = train_super_bpe_tokenizer(texts, num_merges=5, disconnected_merges=3)
        boundless_merges = train_boundless_bpe_tokenizer(texts, num_merges=5)

        assert super_merges == [
            ('a', 'b'),
            (' ', 'c'),
            (' c', 'd'),
            (' ', 'ab'),
            (' cd', ' ab'),
        ]
        assert super_merges == boundless_merges

    def test_default_split(self):
        """Default disconnected_merges should be num_merges // 2."""
        texts = ["the teacher teaches the thick thing"]
        merges = train_super_bpe_tokenizer(texts, num_merges=4)
        assert len(merges) == 4
