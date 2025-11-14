from complex_tokenization.examples.bpe import train_bpe_tokenizer, train_huggingface_tokenizer
from complex_tokenization.examples.utils import text_dataset


class TestBPE:
    def test_basic_train_huggingface_tokenizer(self):
        """Test training HuggingFace tokenizer with expected merges"""
        texts = ["the teacher teaches the thick thing"]
        # Only 2 merges, to avoid needing a tie-breaker
        merges = train_huggingface_tokenizer(texts, num_merges=2)

        expected = [
            ('Ġ', 't'),
            ('h', 'e'),
        ]

        assert merges == expected

    def test_basic_train_complex_tokenizer(self):
        """Test training complex tokenizer with expected merges"""
        texts = ["the teacher teaches the thick thing"]
        # Only 2 merges, to avoid needing a tie-breaker
        merges = train_bpe_tokenizer(texts, num_merges=2)

        expected = [
            (' ', 't'),
            ('h', 'e'),
        ]

        assert merges == expected

    def test_large_train_huggingface_tokenizer(self):
        """Test training HuggingFace tokenizer with expected merges"""
        texts = list(text_dataset(max_samples=10))
        merges = train_huggingface_tokenizer(texts, num_merges=10)

        expected = [
            ("Ġ", "t"),
            ("Ġ", "a"),
            ("o", "n"),
            ("h", "e"),
            ("e", "s"),
            ("e", "r"),
            ("i", "n"),
            ("Ġt", "he"),
            ("e", "d"),
            ("a", "l"),
        ]

        assert merges == expected

    def test_large_train_complex_tokenizer(self):
        """Test training complex tokenizer with expected merges"""
        texts = list(text_dataset(max_samples=10))
        merges = train_bpe_tokenizer(texts, num_merges=10)

        expected = [
            (" ", "t"),
            (" ", "a"),
            ("o", "n"),
            ("h", "e"),
            ("e", "s"),
            ("e", "r"),
            ("i", "n"),
            (" t", "he"),
            ("e", "d"),
            ("a", "l"),
        ]

        assert merges == expected
