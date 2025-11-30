

from complex_tokenization.examples.utils import text_dataset
from complex_tokenization.graphs.settings import GraphSettings
from complex_tokenization.graphs.units import utf8_clusters
from complex_tokenization.graphs.words import words


def train_bne_tokenizer(texts: list[str],
                        n=2,
                        connected=False,
                        units=utf8_clusters,
                        num_merges: int = 10):
    from complex_tokenization.trainer import Trainer

    GraphSettings.ONLY_MINIMAL_MERGES = True  # BNE only merges adjacent tokens
    GraphSettings.MAX_MERGE_SIZE = n  # Maximum number of tokens to merge at a time
    GraphSettings.USE_SINGLETONS = False  # for performance

    graphs = tuple([words(text, connected=connected, units=units) for text in texts])

    trainer = Trainer(graphs=graphs)
    trainer.train(num_merges=num_merges)
    return trainer.get_merges()


if __name__ == "__main__":
    texts = list(text_dataset(max_samples=10))
    print(train_bne_tokenizer(texts, n=4))
