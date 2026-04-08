"""Edge case tests for all tokenizer variants."""

import pytest

from complex_tokenization.tokenizer import (
    BNETokenizer,
    BoundlessBPETokenizer,
    BPETokenizer,
    SuperBPETokenizer,
)

ALL_TOKENIZERS = [
    ("BPE", lambda: BPETokenizer()),
    ("BNE", lambda: BNETokenizer(n=4)),
    ("Boundless", lambda: BoundlessBPETokenizer()),
    ("Super", lambda: SuperBPETokenizer(disconnected_merges=2)),
]


class TestEmptyAndMinimal:
    @pytest.mark.parametrize(("name", "factory"), ALL_TOKENIZERS, ids=[t[0] for t in ALL_TOKENIZERS])
    def test_empty_text(self, name, factory):
        tok = factory()
        merges = tok.train([""], num_merges=10)
        assert merges == []

    @pytest.mark.parametrize(("name", "factory"), ALL_TOKENIZERS, ids=[t[0] for t in ALL_TOKENIZERS])
    def test_single_char(self, name, factory):
        tok = factory()
        merges = tok.train(["a"], num_merges=10)
        assert merges == []

    @pytest.mark.parametrize(("name", "factory"), ALL_TOKENIZERS, ids=[t[0] for t in ALL_TOKENIZERS])
    def test_single_word(self, name, factory):
        tok = factory()
        merges = tok.train(["hello"], num_merges=2)
        assert len(merges) <= 2

    @pytest.mark.parametrize(("name", "factory"), ALL_TOKENIZERS, ids=[t[0] for t in ALL_TOKENIZERS])
    def test_all_same_chars(self, name, factory):
        tok = factory()
        merges = tok.train(["aaaaaaaaaa"], num_merges=5)
        assert len(merges) >= 1
        assert all(t == 'a' for t in merges[0])

    @pytest.mark.parametrize(("name", "factory"), ALL_TOKENIZERS, ids=[t[0] for t in ALL_TOKENIZERS])
    def test_whitespace_only(self, name, factory):
        tok = factory()
        merges = tok.train(["   "], num_merges=5)
        assert len(merges) >= 1

    @pytest.mark.parametrize(("name", "factory"), ALL_TOKENIZERS, ids=[t[0] for t in ALL_TOKENIZERS])
    def test_multiple_empty_texts(self, name, factory):
        tok = factory()
        merges = tok.train(["", "", ""], num_merges=5)
        assert merges == []


class TestUnicodeEdgeCases:
    @pytest.mark.parametrize(("name", "factory"), ALL_TOKENIZERS, ids=[t[0] for t in ALL_TOKENIZERS])
    def test_emoji(self, name, factory):
        tok = factory()
        merges = tok.train(["👋👋👋"], num_merges=3)
        assert len(merges) >= 1

    @pytest.mark.parametrize(("name", "factory"), ALL_TOKENIZERS, ids=[t[0] for t in ALL_TOKENIZERS])
    def test_mixed_scripts(self, name, factory):
        tok = factory()
        merges = tok.train(["hello שלום 你好 hello שלום 你好"], num_merges=5)
        assert len(merges) >= 1

    @pytest.mark.parametrize(("name", "factory"), ALL_TOKENIZERS, ids=[t[0] for t in ALL_TOKENIZERS])
    def test_newlines(self, name, factory):
        tok = factory()
        merges = tok.train(["hello\nworld\nhello\nworld"], num_merges=3)
        assert len(merges) >= 1
