from complex_tokenization.tokenizer import BNETokenizer
from tests.utils import text_dataset


class TestBNE:
    def test_large_train_bne_tokenizer(self):
        texts = list(text_dataset(max_samples=10))
        tok = BNETokenizer(n=4)
        merges = tok.train(texts, num_merges=10)

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
