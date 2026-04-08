"""Test the high-level Tokenizer API."""

import pytest

from complex_tokenization.tokenizer import (
    BNETokenizer,
    BoundlessBPETokenizer,
    BPETokenizer,
    SuperBPETokenizer,
    Tokenizer,
)


class TestTokenizerAPI:
    def test_default_tokenizer(self):
        tok = Tokenizer()
        merges = tok.train(["hello world hello world"], num_merges=3)
        assert len(merges) == 3

    def test_bpe_tokenizer(self):
        tok = BPETokenizer()
        merges = tok.train(["the teacher teaches the thick"], num_merges=2)
        assert all(len(m) == 2 for m in merges)

    def test_bne_tokenizer(self):
        tok = BNETokenizer(n=4)
        merges = tok.train(["the teacher teaches the thick"], num_merges=2)
        assert all(2 <= len(m) <= 4 for m in merges)

    def test_boundless_bpe_tokenizer(self):
        tok = BoundlessBPETokenizer()
        merges = tok.train(["the teacher teaches the thick"], num_merges=2)
        assert all(len(m) == 2 for m in merges)

    def test_super_bpe_tokenizer(self):
        tok = SuperBPETokenizer()
        merges = tok.train(["the teacher teaches the thick"], num_merges=4)
        assert len(merges) == 4

    def test_custom_units(self):
        tok = Tokenizer(units="utf8")
        merges = tok.train(["hello hello"], num_merges=2)
        assert len(merges) == 2

    def test_invalid_units_raises(self):
        with pytest.raises(ValueError, match="Unknown units"):
            Tokenizer(units="invalid")

    def test_callable_units(self):
        from complex_tokenization.graphs.units import utf8
        tok = Tokenizer(units=utf8)
        merges = tok.train(["test test"], num_merges=2)
        assert len(merges) == 2

    def test_get_merges_before_train(self):
        tok = Tokenizer()
        assert tok.get_merges() == []

    def test_super_bpe_phase1_matches_bpe(self):
        texts = ["the teacher teaches the thick thing"] * 3
        bpe = BPETokenizer()
        bpe_merges = bpe.train(texts, num_merges=5)
        super_bpe = SuperBPETokenizer(disconnected_merges=5)
        super_merges = super_bpe.train(texts, num_merges=10)
        assert super_merges[:5] == bpe_merges

    def test_custom_pretokenizer_regex(self):
        from tokenizers import Regex
        from tokenizers.pre_tokenizers import Split
        tok = BPETokenizer(pretokenizer=Split(Regex(r"\w+|\S"), behavior="isolated"))
        merges = tok.train(["hello hello hello"], num_merges=2)
        assert len(merges) >= 1

    def test_custom_pretokenizer_whitespace(self):
        from tokenizers.pre_tokenizers import Whitespace
        tok = BPETokenizer(pretokenizer=Whitespace())
        merges = tok.train(["ab ab ab cd cd cd"], num_merges=2)
        assert len(merges) >= 1

    def test_different_pretokenizer_different_merges(self):
        from tokenizers.pre_tokenizers import Whitespace
        texts = ["hello-world hello-world hello-world"]
        default_merges = BPETokenizer().train(texts, num_merges=5)
        whitespace_merges = BPETokenizer(pretokenizer=Whitespace()).train(texts, num_merges=5)
        assert default_merges != whitespace_merges
