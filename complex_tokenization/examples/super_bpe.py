from complex_tokenization.examples.utils import text_dataset
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

    GraphSettings.ONLY_MINIMAL_MERGES = True
    GraphSettings.MAX_MERGE_SIZE = 2
    GraphSettings.USE_SINGLETONS = False

    graphs = tuple([words(text, connected=False, units=utf8_clusters) for text in texts])
    trainer = Trainer(graphs=graphs)
    trainer.train(num_merges=disconnected_merges)
    phase1_merges = list(trainer.merges)

    connected_graphs = tuple([words(text, connected=True, units=utf8_clusters) for text in texts])
    trainer_phase2 = Trainer(graphs=connected_graphs)
    for token, merge in phase1_merges:
        trainer_phase2.graph = trainer_phase2.graph.merge(token, merge)
    trainer_phase2.merges = phase1_merges

    remaining = num_merges - len(phase1_merges)
    if remaining > 0:
        trainer_phase2.train(num_merges=num_merges)

    return trainer_phase2.get_merges()


if __name__ == "__main__":
    texts = list(text_dataset(max_samples=10))
    print(train_super_bpe_tokenizer(texts))
