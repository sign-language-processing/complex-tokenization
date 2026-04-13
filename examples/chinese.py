"""Chinese tokenizer example: Standard BPE vs BPE + Chinese IDS decomposition.

Trains on Chinese Wikipedia, plots token count reduction over merges.
"""

from itertools import islice

from complex_tokenization_fast import BPETokenizer
from complex_tokenization_fast.graphs.units import register_script
from complex_tokenization_fast.languages.chinese.graph import chinese_character_to_graph
from datasets import load_dataset

NUM_MERGES = 6000


def load_texts(n=100):
    ds = load_dataset(
        "wikimedia/wikipedia", "20231101.zh",
        split="train", streaming=True,
    )
    return [row["text"] for row in islice(ds, n) if row["text"]]


def train_and_count(tok, texts, num_merges, sample_every=1):
    trainer = tok.make_trainer(texts)
    xs, ys = trainer.train_with_counts(num_merges, sample_every)
    return list(xs), list(ys)


def main():
    print("Loading data...")
    texts = load_texts()
    sample_every = max(1, NUM_MERGES // 100)

    print(f"Training Standard BPE ({NUM_MERGES} merges)...")
    std_x, std_y = train_and_count(BPETokenizer(), texts, NUM_MERGES, sample_every)

    print(f"Training BPE + Chinese IDS ({NUM_MERGES} merges)...")
    register_script("Han", chinese_character_to_graph)
    cn_x, cn_y = train_and_count(BPETokenizer(), texts, NUM_MERGES, sample_every)

    print(f"\n{'Merge':>6s}  {'Std BPE':>10s}  {'BPE+IDS':>10s}")
    for i in range(min(len(std_x), len(cn_x))):
        print(f"{std_x[i]:>6d}  {std_y[i]:>10,}  {cn_y[i]:>10,}")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(std_x, std_y, label="Standard BPE")
        ax.plot(cn_x, cn_y, label="BPE + Chinese IDS")
        ax.set_xlabel("Merges")
        ax.set_yscale("log")
        ax.set_ylabel("Token count (log scale)")
        ax.set_title(f"Token count vs merges ({len(texts)} docs)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig("examples/chinese_merges.png", dpi=150)
        print("\nSaved examples/chinese_merges.png")
    except ImportError:
        print("\npip install matplotlib for the plot")


if __name__ == "__main__":
    main()
