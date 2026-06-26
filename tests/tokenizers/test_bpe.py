from complex_tokenization.graphs.settings import GraphSettings
from complex_tokenization.tokenizer import BPETokenizer
from tests.utils import text_dataset, train_huggingface_tokenizer


class TestBPE:
    def test_basic_train_huggingface_tokenizer(self):
        texts = ["the teacher teaches the thick thing"]
        merges = train_huggingface_tokenizer(texts, num_merges=2)
        expected = [('Ġ', 't'), ('h', 'e')]
        assert merges == expected

    def test_basic_train_complex_tokenizer(self):
        texts = ["the teacher teaches the thick thing"]
        tok = BPETokenizer()
        merges = tok.train(texts, num_merges=2)
        expected = [(' ', 't'), ('h', 'e')]
        assert merges == expected

    def test_large_train_huggingface_tokenizer(self):
        texts = list(text_dataset(max_samples=10))
        merges = train_huggingface_tokenizer(texts, num_merges=10)
        expected = [
            ("Ġ", "t"), ("Ġ", "a"), ("o", "n"), ("h", "e"), ("e", "s"),
            ("e", "r"), ("i", "n"), ("Ġt", "he"), ("e", "d"), ("a", "l"),
        ]
        assert merges == expected

    def test_large_train_complex_tokenizer(self):
        texts = list(text_dataset(max_samples=10))
        tok = BPETokenizer()
        merges = tok.train(texts, num_merges=10)
        expected = [
            (" ", "t"), (" ", "a"), ("o", "n"), ("h", "e"), ("e", "s"),
            ("e", "r"), ("i", "n"), (" t", "he"), ("e", "d"), ("a", "l"),
        ]
        assert merges == expected

    def test_memoization_matches_unmemoized(self):
        texts = list(text_dataset(max_samples=10))
        GraphSettings.TRADE_MEMORY_FOR_SPEED = False
        plain = BPETokenizer().train(texts, num_merges=20)
        GraphSettings.TRADE_MEMORY_FOR_SPEED = True
        memoized = BPETokenizer().train(texts, num_merges=20)
        assert memoized == plain
