from complex_tokenization.examples.bne import train_bne_tokenizer
from complex_tokenization.examples.utils import text_dataset


class TestBNE:
    def test_large_train_bne_tokenizer(self):
        """Test training BNE tokenizer with n=4 and expected merges"""
        texts = list(text_dataset(max_samples=10))
        merges = train_bne_tokenizer(texts, n=4, num_merges=10)

        expected = [
            (' ', 't', 'h', 'e'),
            (' ', 'a'),
            ('i', 'o', 'n'),
            ('e', 'r'),
            (' ', 'g', 'a', 'm'),
            (' ', 'V', 'a', 'l'),
            ('e', 's'),
            ('i', 'n'),
            (' Val', 'k', 'y', 'r'),
            ('e', 'd')
        ]

        assert merges == expected