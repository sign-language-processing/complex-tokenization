from complex_tokenization.tokenizer import BoundlessBPETokenizer, BPETokenizer


class TestBoundlessBPE:
    def test_basic_boundless_bpe(self):
        texts = ["the teacher teaches the thick thing"]
        tok = BoundlessBPETokenizer()
        merges = tok.train(texts, num_merges=2)
        assert len(merges) == 2

    def test_boundless_extends_bpe_with_cross_word_merges(self):
        """BPE exhausts intra-word merges; boundless continues across words."""
        texts = ["ab cd ab cd ab cd"]

        bpe_merges = BPETokenizer().train(texts, num_merges=5)
        boundless_merges = BoundlessBPETokenizer().train(texts, num_merges=5)

        assert bpe_merges == [
            ('a', 'b'), (' ', 'c'), (' c', 'd'), (' ', 'ab'),
        ]
        assert boundless_merges == [
            ('a', 'b'), (' ', 'c'), (' c', 'd'), (' ', 'ab'), (' cd', ' ab'),
        ]
        assert boundless_merges[:len(bpe_merges)] == bpe_merges
        assert len(boundless_merges) > len(bpe_merges)
