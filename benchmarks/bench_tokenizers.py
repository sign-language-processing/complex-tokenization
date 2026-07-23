"""Benchmark all tokenizer variants, in both implementations, against HuggingFace.

Usage (from the repo root):
    python benchmarks/bench_tokenizers.py                     # quick run
    python benchmarks/bench_tokenizers.py --samples 0 --merges 500   # full wikitext-2 train split

Each case runs in its own subprocess so peak memory is the OS-level RSS
high-water mark of that case alone (ru_maxrss covers Rust allocations too,
unlike tracemalloc). Digests are md5 of the merge list, for checking that
implementations produce identical output.
"""

import argparse
import hashlib
import resource
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # for tests.utils

CASES = ["BPE", "BNE n=4", "Boundless BPE", "Super BPE"]
IMPLS = ["HuggingFace", "reference (Python)", "fast (Rust)"]


def train(case, impl, texts, num_merges):
    if impl == "HuggingFace":
        from tests.utils import train_huggingface_tokenizer
        return train_huggingface_tokenizer(texts, num_merges=num_merges)
    if impl == "reference (Python)":
        import complex_tokenization.tokenizer as module
    else:
        import complex_tokenization_fast.tokenizer as module
    tokenizer = {
        "BPE": module.BPETokenizer,
        "BNE n=4": lambda: module.BNETokenizer(n=4),
        "Boundless BPE": module.BoundlessBPETokenizer,
        "Super BPE": module.SuperBPETokenizer,
    }[case]()
    return tokenizer.train(texts, num_merges=num_merges)


def run_case(case, impl, samples, num_merges):
    from tests.utils import text_dataset
    texts = [t for t in text_dataset(max_samples=samples or None) if t.strip()]

    start = time.perf_counter()
    merges = train(case, impl, texts, num_merges)
    elapsed = time.perf_counter() - start

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak / (1e6 if sys.platform == "darwin" else 1e3)  # bytes on macOS, KB on Linux
    digest = hashlib.md5(repr(list(merges)).encode()).hexdigest()[:10]
    print(f"| {case} | {impl} | {elapsed:.3f}s | {peak_mb:.0f} MB | {len(merges)} | `{digest}` |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50,
                        help="dataset rows to use; 0 = the full wikitext-2 train split (~2M words)")
    parser.add_argument("--merges", type=int, default=100)
    parser.add_argument("--case", choices=CASES, help=argparse.SUPPRESS)  # subprocess mode
    parser.add_argument("--impl", choices=IMPLS, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.case:
        run_case(args.case, args.impl, args.samples, args.merges)
        sys.exit(0)

    try:
        import complex_tokenization_fast  # noqa: F401
        impls = IMPLS
    except ImportError:
        impls = IMPLS[:-1]

    rows = []
    for impl in impls:
        for case in CASES if impl != "HuggingFace" else ["BPE"]:
            result = subprocess.run(
                [sys.executable, __file__, "--case", case, "--impl", impl,
                 "--samples", str(args.samples), "--merges", str(args.merges)],
                capture_output=True, text=True, cwd=Path(__file__).parent.parent,
            )
            row = [line for line in result.stdout.splitlines() if line.startswith("|")]
            rows.extend(row or [f"| {case} | {impl} | failed | | | see stderr |"])
            if not row:
                print(result.stderr, file=sys.stderr)

    from tests.utils import text_dataset
    texts = [t for t in text_dataset(max_samples=args.samples or None) if t.strip()]
    print(f"Corpus: {len(texts):,} documents, {sum(len(t) for t in texts):,} characters"
          f" | {args.merges} merges\n")
    print("| Tokenizer | Implementation | Time | Peak RSS | Merges | Digest |")
    print("|---|---|--:|--:|--:|---|")
    print("\n".join(rows))
