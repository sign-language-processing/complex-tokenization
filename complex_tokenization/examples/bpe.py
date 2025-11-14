import json

from tokenizers import Tokenizer
from tqdm import tqdm

from complex_tokenization.examples.bne import train_bne_tokenizer
from complex_tokenization.examples.utils import text_dataset
from complex_tokenization.graph import GraphSettings, UnconnectedGraphs
from complex_tokenization.graphs.words import words


def get_tokenizer_merges(tokenizer: Tokenizer):
    backend = tokenizer.backend_tokenizer
    data = json.loads(backend.to_str())
    return [tuple(m) for m in data["model"]["merges"]]


def train_huggingface_tokenizer(texts: list[str], num_merges: int = 10):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

    new_tokenizer = tokenizer.train_new_from_iterator(texts, 256 + 21 + num_merges)
    return get_tokenizer_merges(new_tokenizer)


def train_bpe_tokenizer(texts: list[str], num_merges: int = 10):
    # BPE can only merge 2 tokens at a time
    return train_bne_tokenizer(texts, n=2, num_merges=num_merges)


if __name__ == "__main__":
    texts = list(text_dataset(max_samples=10))
    print(train_bpe_tokenizer(texts))
    print(train_huggingface_tokenizer(texts))
