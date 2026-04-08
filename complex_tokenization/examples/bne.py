

from functools import reduce

from complex_tokenization.examples.utils import text_dataset
from complex_tokenization.graph import Node
from complex_tokenization.graphs.settings import GraphSettings
from complex_tokenization.graphs.units import utf8_clusters
from complex_tokenization.graphs.words import words


def train_bne_tokenizer(texts: list[str],
                        n=2,
                        connected=False,
                        units=utf8_clusters,
                        num_merges: int = 10,
                        known_merges: list[tuple[str, ...]] | None = None):
    from complex_tokenization.trainer import Trainer

    GraphSettings.ONLY_MINIMAL_MERGES = True
    GraphSettings.MAX_MERGE_SIZE = n

    graphs = tuple(words(text, connected=connected, units=units) for text in texts)

    trainer = Trainer(graphs=graphs)

    if known_merges:
        for merge_strs in known_merges:
            nodes = tuple(Node(value=s.encode("utf-8")) for s in merge_strs)
            token = reduce(lambda a, b: a + b, nodes)
            trainer.graph = trainer.graph.merge(token, nodes)
            trainer.merges.append((token, nodes))

    trainer.train(num_merges=num_merges)
    return trainer.get_merges()


if __name__ == "__main__":
    texts = list(text_dataset(max_samples=10))
    print(train_bne_tokenizer(texts, n=4))
