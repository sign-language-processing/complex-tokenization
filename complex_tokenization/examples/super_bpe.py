from complex_tokenization.examples.bne import train_bne_tokenizer
from complex_tokenization.examples.utils import text_dataset


def train_super_bpe_tokenizer(texts: list[str],
                              num_merges: int = 10,
                              disconnected_merges: int | None = None):
    """Train with disconnected merges first, then switch to connected.

    Phase 1: Train BPE with word boundaries (connected=False) to learn
    intra-word patterns like common subwords.
    Phase 2: Switch to connected=True to learn cross-word patterns like
    frequent word combinations, seeded with phase 1 merges.
    """
    if disconnected_merges is None:
        disconnected_merges = num_merges // 2

    phase1 = train_bne_tokenizer(texts, n=2, connected=False, num_merges=disconnected_merges)
    return train_bne_tokenizer(texts, n=2, connected=True, num_merges=num_merges, known_merges=phase1)


if __name__ == "__main__":
    texts = list(text_dataset(max_samples=10))
    print(train_super_bpe_tokenizer(texts))
