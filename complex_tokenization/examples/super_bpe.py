from functools import reduce

from complex_tokenization.examples.bne import train_bne_tokenizer
from complex_tokenization.examples.utils import text_dataset
from complex_tokenization.graph import Node
from complex_tokenization.graphs.settings import GraphSettings
from complex_tokenization.graphs.units import utf8_clusters
from complex_tokenization.graphs.words import words
from complex_tokenization.trainer import Trainer


def train_super_bpe_tokenizer(texts: list[str],
                              num_merges: int = 10,
                              disconnected_merges: int | None = None):
    """Train with disconnected merges first, then switch to connected.

    Phase 1: Train BPE with word boundaries (connected=False) to learn
    intra-word patterns like common subwords.
    Phase 2: Switch to connected=True to learn cross-word patterns like
    frequent word combinations.
    """
    if disconnected_merges is None:
        disconnected_merges = num_merges // 2

    # Phase 1: standard BPE with word boundaries
    phase1_merges = train_bne_tokenizer(texts, n=2, connected=False,
                                        units=utf8_clusters, num_merges=disconnected_merges)

    # Phase 2: rebuild graphs without word boundaries, then replay phase 1
    # merges on the new connected graph so the trainer continues from where
    # phase 1 left off — but now cross-word pairs are visible.
    GraphSettings.ONLY_MINIMAL_MERGES = True
    GraphSettings.MAX_MERGE_SIZE = 2
    GraphSettings.USE_SINGLETONS = False

    connected_graphs = tuple(words(text, connected=True, units=utf8_clusters) for text in texts)
    trainer = Trainer(graphs=connected_graphs)

    for merge_strs in phase1_merges:
        nodes_for_merge = tuple(Node(value=s.encode("utf-8")) for s in merge_strs)
        token = reduce(lambda a, b: a + b, nodes_for_merge)
        trainer.graph = trainer.graph.merge(token, nodes_for_merge)
        trainer.merges.append((token, nodes_for_merge))

    remaining = num_merges - len(trainer.merges)
    if remaining > 0:
        trainer.train(num_merges=num_merges)

    return trainer.get_merges()


if __name__ == "__main__":
    texts = list(text_dataset(max_samples=10))
    print(train_super_bpe_tokenizer(texts))
