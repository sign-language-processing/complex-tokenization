"""Benchmark Chinese BPE training at various scales."""
import time
from itertools import islice

from datasets import load_dataset

from complex_tokenization import BPETokenizer
from complex_tokenization.graphs.units import register_script
from complex_tokenization.languages.chinese.graph import chinese_character_to_graph


def load_texts(n):
    ds = load_dataset(
        "fjcanyue/wikipedia-zh-cn",
        data_files="wikipedia-zh-cn-20260201.json",
        split="train",
        streaming=True,
    )
    return [row["text"][:500] for row in islice(ds, n) if row["text"]]


def bench(texts, num_merges=10, label=""):
    t0 = time.perf_counter()
    tok = BPETokenizer()
    merges = tok.train(texts, num_merges=num_merges)
    elapsed = time.perf_counter() - t0
    per_merge = elapsed / num_merges if num_merges else 0
    print(f"  {label:30s} {elapsed:7.3f}s  ({per_merge:.4f}s/merge, {len(merges)} merges)")
    return elapsed


if __name__ == "__main__":
    register_script("Han", chinese_character_to_graph)

    for n in [10, 50, 100]:
        texts = load_texts(n)
        print(f"\n--- {len(texts)} docs ---")
        bench(texts, num_merges=10, label=f"{len(texts)} docs, 10 merges")
        bench(texts, num_merges=50, label=f"{len(texts)} docs, 50 merges")
