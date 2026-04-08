import json

from datasets import load_dataset
from tokenizers import Tokenizer


def text_dataset(max_samples=None,
                 dataset="Salesforce/wikitext",
                 dataset_config="wikitext-2-raw-v1"):
    dataset = load_dataset(dataset, dataset_config, streaming=True, split="train")
    if max_samples is not None:
        dataset = dataset.take(max_samples)
    return (sample["text"] for sample in dataset)


def get_tokenizer_merges(tokenizer: Tokenizer):
    backend = tokenizer.backend_tokenizer
    data = json.loads(backend.to_str())
    return [tuple(m) for m in data["model"]["merges"]]


def train_huggingface_tokenizer(texts: list[str], num_merges: int = 10):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

    new_tokenizer = tokenizer.train_new_from_iterator(texts, 256 + 21 + num_merges)
    return get_tokenizer_merges(new_tokenizer)
