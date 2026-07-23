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


def bench(rows, name, impl, train):
    tracemalloc.start()
    start = time.perf_counter()
    merges = train()
    elapsed = time.perf_counter() - start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    digest = hashlib.md5(repr(list(merges)).encode()).hexdigest()[:10]
    rows.append(f"| {name} | {impl} | {elapsed:.3f}s | {peak_mem / 1024 / 1024:.2f} MB "
                f"| {len(merges)} | `{digest}` |")


def run_benchmarks(rows, module, impl, texts, num_merges):
    bench(rows, "BPE", impl, lambda: module.BPETokenizer().train(texts, num_merges=num_merges))
    bench(rows, "BNE n=4", impl, lambda: module.BNETokenizer(n=4).train(texts, num_merges=num_merges))
    bench(rows, "Boundless BPE", impl, lambda: module.BoundlessBPETokenizer().train(texts, num_merges=num_merges))
    bench(rows, "Super BPE", impl, lambda: module.SuperBPETokenizer().train(texts, num_merges=num_merges))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50,
                        help="dataset rows to use; 0 = the full wikitext-2 train split (~2M words)")
    parser.add_argument("--merges", type=int, default=100)
    args = parser.parse_args()

    texts = [t for t in text_dataset(max_samples=args.samples or None) if t.strip()]

    # Warm lazy imports and module-level caches, so the first row's peak
    # memory doesn't absorb one-time allocations the later rows reuse.
    train_huggingface_tokenizer(["warm up"], num_merges=1)
    reference.BPETokenizer().train(["warm up"], num_merges=1)
    if fast is not None:
        fast.BPETokenizer().train(["warm up"], num_merges=1)

    # Rows are collected and the table printed at the end, because library
    # progress bars (HF tokenizers) write straight to the OS fd mid-run.
    rows = []
    bench(rows, "BPE", "HuggingFace", lambda: train_huggingface_tokenizer(texts, num_merges=args.merges))
    run_benchmarks(rows, reference, "reference (Python)", texts, args.merges)
    if fast is not None:
        run_benchmarks(rows, fast, "fast (Rust)", texts, args.merges)

    print(f"\nCorpus: {len(texts):,} documents, {sum(len(t) for t in texts):,} characters"
          f" | {args.merges} merges\n")
    print("| Tokenizer | Implementation | Time | Peak mem | Merges | Digest |")
    print("|---|---|--:|--:|--:|---|")
    print("\n".join(rows))
