from tokenizers import Tokenizer
from transformers import AutoTokenizer
import json

from complex_tokenization.examples.utils import text_dataset
from complex_tokenization.graph import GraphSettings


def get_tokenizer_merges(tokenizer: Tokenizer):
    backend = tokenizer.backend_tokenizer
    data = json.loads(backend.to_str())
    return [tuple(m) for m in data["model"]["merges"]]


def train_huggingface_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

    texts = text_dataset(max_samples=10)
    new_tokenizer = tokenizer.train_new_from_iterator(texts, 256 + 30)
    return get_tokenizer_merges(new_tokenizer)


def train_complex_tokenizer():
    from complex_tokenization.graph import text_to_graph
    from complex_tokenization.trainer import Trainer

    GraphSettings.ONLY_MINIMAL_MERGES = True  # BPE only merges adjacent tokens
    GraphSettings.MAX_MERGE_SIZE = 2  # BPE can only merge 2 tokens at a time
    GraphSettings.USE_SINGLETONS = True  # for performance

    texts = text_dataset(max_samples=10)
    graphs = (text_to_graph(text) for text in texts)

    # TODO: replace NodeSequence-s with a single "UnconnectedNodes" of all nodes inside

    trainer = Trainer()
    trainer.train(graphs, num_merges=10)
    return list(trainer.get_merges())


if __name__ == "__main__":
    print(train_huggingface_tokenizer())
