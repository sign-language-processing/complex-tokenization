from complex_tokenization.tokenizer import BoundlessBPETokenizer, BPETokenizer, SuperBPETokenizer


class TestSuperBPE:
    def test_basic_super_bpe(self):
        texts = ["the teacher teaches the thick thing"]
        merges = SuperBPETokenizer(disconnected_merges=2).train(texts, num_merges=4)
        assert len(merges) == 4

    def test_phase1_matches_bpe(self):
        texts = ["ab cd ab cd ab cd"] * 3
        bpe_merges = BPETokenizer().train(texts, num_merges=3)
        super_merges = SuperBPETokenizer(disconnected_merges=3).train(texts, num_merges=5)
        assert super_merges[:3] == bpe_merges

    def test_super_bpe_differs_from_boundless(self):
        """Super BPE prioritizes intra-word merges; boundless picks by global frequency."""
        texts = ["ab ac ab ac ab ac abcdefghik"]

        boundless = BoundlessBPETokenizer().train(texts, num_merges=10)
        super_bpe = SuperBPETokenizer(disconnected_merges=8).train(texts, num_merges=10)

        assert boundless == [
            (' ', 'a'), (' a', 'c'), (' a', 'b'), (' ac', ' ab'),
            ('a', 'b'), ('ab', ' ac ab'), ('ab ac ab', ' ac ab'),
            ('ab ac ab ac ab', ' ac'), (' ab', 'c'), (' abc', 'd'),
        ]
        assert super_bpe == [
            (' ', 'a'), (' a', 'c'), (' a', 'b'), ('a', 'b'),
            (' ab', 'c'), (' abc', 'd'), (' abcd', 'e'), (' abcde', 'f'),
            (' ac', ' ab'), ('ab', ' ac ab'),
        ]

        assert boundless[3] == (' ac', ' ab')
        assert super_bpe[3] == ('a', 'b')

    def test_default_split(self):
        texts = ["the teacher teaches the thick thing"]
        merges = SuperBPETokenizer().train(texts, num_merges=4)
        assert len(merges) == 4
