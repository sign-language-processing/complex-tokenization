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

    def test_super_bpe_differs_from_boundless(self):
        """Super BPE prioritizes intra-word merges; boundless picks by global frequency.

        With 'ab ac ab ac abcdefghik':
        - Boundless merges cross-word 'ab ac' early (high frequency pair)
        - Super forces intra-word merges first (consuming the long word),
          then does cross-word merges later
        """
        texts = ["ab ac ab ac abcdefghik"]

        boundless = train_boundless_bpe_tokenizer(texts, num_merges=10)
        super_bpe = train_super_bpe_tokenizer(texts, num_merges=10, disconnected_merges=8)

        assert boundless == [
            (' ', 'a'),
            (' a', 'c'),
            (' a', 'b'),
            ('a', 'b'),
            ('ab', ' ac'),
            ('ab ac', ' ab'),
            (' ab', 'c'),
            (' abc', 'd'),
            (' abcd', 'e'),
            (' abcde', 'f'),
        ]

        assert super_bpe == [
            (' ', 'a'),
            (' a', 'c'),
            (' a', 'b'),
            ('a', 'b'),
            (' ab', 'c'),
            (' abc', 'd'),
            (' abcd', 'e'),
            (' abcde', 'f'),
            ('ab', ' ac'),
            ('ab ac', ' ab'),
        ]

        assert boundless[4] == ('ab', ' ac')
        assert super_bpe[4] == (' ab', 'c')

    def test_default_split(self):
        """Default disconnected_merges should be num_merges // 2."""
        texts = ["the teacher teaches the thick thing"]
        merges = train_super_bpe_tokenizer(texts, num_merges=4)
        assert len(merges) == 4
