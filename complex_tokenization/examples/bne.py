import json

from tokenizers import Tokenizer
from tqdm import tqdm

from complex_tokenization.examples.utils import text_dataset
from complex_tokenization.graph import GraphSettings, UnconnectedGraphs
from complex_tokenization.graphs.words import words


def train_bne_tokenizer(texts: list[str], n=2, num_merges: int = 10, connected=False):
    from complex_tokenization.trainer import Trainer

    GraphSettings.ONLY_MINIMAL_MERGES = True  # BPE only merges adjacent tokens
    GraphSettings.MAX_MERGE_SIZE = n  # How many tokens to merge at a time
    GraphSettings.USE_SINGLETONS = False  # for performance

    graph = UnconnectedGraphs([words(text, connected=connected) for text in tqdm(texts)])

    trainer = Trainer(graph=graph)
    trainer.train(num_merges=num_merges)
    return trainer.get_merges()


if __name__ == "__main__":
    texts = list(text_dataset(max_samples=10))
    print(train_bne_tokenizer(texts, n=4))
