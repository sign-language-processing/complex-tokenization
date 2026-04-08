from complex_tokenization.examples.bne import train_bne_tokenizer
from complex_tokenization.examples.utils import text_dataset


def train_boundless_bpe_tokenizer(texts: list[str], num_merges: int = 10):
    return train_bne_tokenizer(texts, n=2, connected=True, num_merges=num_merges)


if __name__ == "__main__":
    texts = list(text_dataset(max_samples=10))
    print(train_boundless_bpe_tokenizer(texts))
