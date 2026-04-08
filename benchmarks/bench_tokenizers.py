"""Benchmark all tokenizer variants against each other and HuggingFace."""

import time
import tracemalloc

from complex_tokenization.examples.bne import train_bne_tokenizer
from complex_tokenization.examples.boundless_bpe import train_boundless_bpe_tokenizer
from complex_tokenization.examples.bpe import train_bpe_tokenizer, train_huggingface_tokenizer
from complex_tokenization.examples.super_bpe import train_super_bpe_tokenizer
from complex_tokenization.examples.utils import text_dataset


def bench(name, fn, *args, **kwargs):
    tracemalloc.start()
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"{name:30s}  {elapsed:8.3f}s  {peak_mem / 1024 / 1024:8.2f} MB  merges={len(result)}")
    return result, elapsed, peak_mem


def run_benchmarks(num_samples=10, num_merges=50):
    print(f"\n{'='*70}")
    print(f"Benchmark: {num_samples} samples, {num_merges} merges")
    print(f"{'='*70}")
    texts = list(text_dataset(max_samples=num_samples))
    total_chars = sum(len(t) for t in texts)
    print(f"Total text: {total_chars:,} characters\n")

    print(f"{'Tokenizer':30s}  {'Time':>8s}  {'Peak Mem':>8s}     {'Merges':>6s}")
    print("-" * 70)

    bench("HuggingFace BPE", train_huggingface_tokenizer, texts, num_merges=num_merges)
    bench("BPE (ours)", train_bpe_tokenizer, texts, num_merges=num_merges)
    bench("BNE n=4 (ours)", train_bne_tokenizer, texts, n=4, num_merges=num_merges)
    bench("Boundless BPE (ours)", train_boundless_bpe_tokenizer, texts, num_merges=num_merges)
    bench("Super BPE (ours)", train_super_bpe_tokenizer, texts, num_merges=num_merges)


if __name__ == "__main__":
    for samples in [10, 50]:
        for merges in [50, 100]:
            run_benchmarks(num_samples=samples, num_merges=merges)
