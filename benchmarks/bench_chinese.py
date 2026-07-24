"""Benchmark Chinese BPE training at various scales."""
import hashlib
import time
from itertools import islice

import complex_tokenization_fast
from datasets import load_dataset

import complex_tokenization

IMPLS = {"reference (Python)": complex_tokenization, "fast (Rust)": complex_tokenization_fast}


def load_texts(n):
    ds = load_dataset(
        "fjcanyue/wikipedia-zh-cn",
        data_files="wikipedia-zh-cn-20260201.json",
        split="train",
        streaming=True,
    )
    return [row["text"][:500] for row in islice(ds, n) if row["text"]]


def bench(module, impl, texts, num_merges=10):
    # Each implementation has its own script registry and BPETokenizer.
    from importlib import import_module
    pkg = module.__name__
    units = import_module(f"{pkg}.graphs.units")
    chinese = import_module(f"{pkg}.languages.chinese.graph")
    units.register_script("Han", chinese.chinese_character_to_graph)

    t0 = time.perf_counter()
    tok = module.BPETokenizer()
    merges = tok.train(texts, num_merges=num_merges)
    elapsed = time.perf_counter() - t0
    digest = hashlib.md5(repr(list(merges)).encode()).hexdigest()[:10]
    print(f"  {impl:22s} {elapsed:7.3f}s  ({elapsed / num_merges:.4f}s/merge, "
          f"{len(merges)} merges, digest={digest})")
    return elapsed


if __name__ == "__main__":
    for n in [10, 50, 100]:
        texts = load_texts(n)
        for num_merges in [10, 50]:
            print(f"\n--- {len(texts)} docs, {num_merges} merges ---")
            for impl, module in IMPLS.items():
                bench(module, impl, texts, num_merges=num_merges)
