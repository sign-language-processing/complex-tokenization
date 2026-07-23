"""Benchmark all tokenizer variants, in both implementations, against HuggingFace.

Usage (from the repo root):
    python benchmarks/bench_tokenizers.py                     # quick run
    python benchmarks/bench_tokenizers.py --samples 0 --merges 500   # full wikitext-2 train split

Peak memory is tracemalloc, which only tracks Python allocations — for the
fast (Rust) implementation it excludes Rust-side memory, so compare memory
between reference rows only. Digests are md5 of the merge list, for checking
that implementations produce identical output.
"""

import argparse
import hashlib
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # for tests.utils

import complex_tokenization.tokenizer as reference
from tests.utils import text_dataset, train_huggingface_tokenizer

try:
    import complex_tokenization_fast.tokenizer as fast
except ImportError:
    fast = None


def bench(name, train):
    tracemalloc.start()
    start = time.perf_counter()
    merges = train()
    elapsed = time.perf_counter() - start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    digest = hashlib.md5(repr(list(merges)).encode()).hexdigest()[:10]
    print(f"{name:30s}  {elapsed:8.3f}s  {peak_mem / 1024 / 1024:8.2f} MB  "
          f"merges={len(merges):4d}  digest={digest}")


def run_benchmarks(module, label, texts, num_merges):
    print(f"\n[{label}]")
    bench("BPE", lambda: module.BPETokenizer().train(texts, num_merges=num_merges))
    bench("BNE n=4", lambda: module.BNETokenizer(n=4).train(texts, num_merges=num_merges))
    bench("Boundless BPE", lambda: module.BoundlessBPETokenizer().train(texts, num_merges=num_merges))
    bench("Super BPE", lambda: module.SuperBPETokenizer().train(texts, num_merges=num_merges))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50,
                        help="dataset rows to use; 0 = the full wikitext-2 train split (~2M words)")
    parser.add_argument("--merges", type=int, default=100)
    args = parser.parse_args()

    texts = [t for t in text_dataset(max_samples=args.samples or None) if t.strip()]
    print(f"Corpus: {len(texts):,} documents, {sum(len(t) for t in texts):,} characters"
          f" | {args.merges} merges")
    print(f"{'Tokenizer':30s}  {'Time':>9s}  {'Peak Mem':>11s}")

    bench("HuggingFace BPE", lambda: train_huggingface_tokenizer(texts, num_merges=args.merges))
    run_benchmarks(reference, "reference (Python)", texts, args.merges)
    if fast is not None:
        run_benchmarks(fast, "fast (Rust)", texts, args.merges)
