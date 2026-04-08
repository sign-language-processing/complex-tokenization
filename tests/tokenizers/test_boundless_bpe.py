from complex_tokenization.examples.boundless_bpe import train_boundless_bpe_tokenizer
from complex_tokenization.examples.bpe import train_bpe_tokenizer
from complex_tokenization.examples.utils import text_dataset


class TestBoundlessBPE:
    def test_basic_boundless_bpe(self):
        texts = ["the teacher teaches the thick thing"]
        merges = train_boundless_bpe_tokenizer(texts, num_merges=2)
        assert len(merges) == 2

    def test_boundless_bpe_can_merge_across_words(self):
        """Boundless BPE should be able to merge tokens across word boundaries."""
        texts = ["ab cd ab cd ab cd"]
        merges_bounded = train_bpe_tokenizer(texts, num_merges=5)
        merges_boundless = train_boundless_bpe_tokenizer(texts, num_merges=5)
        assert merges_bounded != merges_boundless

    def test_boundless_bpe_starts_same_as_bpe(self):
        """First merges should be similar since intra-word pairs dominate."""
        texts = list(text_dataset(max_samples=10))
        merges_bpe = train_bpe_tokenizer(texts, num_merges=3)
        merges_boundless = train_boundless_bpe_tokenizer(texts, num_merges=3)
        assert merges_bpe[0] == merges_boundless[0]

    def test_large_boundless_bpe(self):
        texts = list(text_dataset(max_samples=10))
        merges = train_boundless_bpe_tokenizer(texts, num_merges=10)
        assert len(merges) == 10
        for merge in merges:
            assert len(merge) == 2
