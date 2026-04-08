"""Compare our BPE implementations against a simple reference implementation.

The reference implementation is a straightforward BPE that operates on
byte sequences, without any graph abstraction. It serves as ground truth.
"""

from collections import Counter

import regex

from complex_tokenization.fast_bpe_trainer import FastBPETrainer
from complex_tokenization.tokenizer import BoundlessBPETokenizer, BPETokenizer

PATTERN = (
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    "|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    "|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
)


def _apply_merge_to_freqs(word_freqs, best_pair):
    a, b = best_pair
    merged = a + b
    new_freqs = {}
    for word, freq in word_freqs.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_freqs[tuple(new_word)] = new_freqs.get(tuple(new_word), 0) + freq
    return new_freqs


def _run_bpe(word_freqs, num_merges):
    merges = []
    for _ in range(num_merges):
        pair_counts: dict[tuple[bytes, bytes], int] = Counter()
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair_counts[(word[i], word[i + 1])] += freq

        if not pair_counts:
            break

        best = max(pair_counts, key=pair_counts.get)
        word_freqs = _apply_merge_to_freqs(word_freqs, best)
        a, b = best
        merges.append((a.decode("utf-8", errors="replace"),
                       b.decode("utf-8", errors="replace")))
    return merges


def reference_bpe(texts, num_merges):
    word_freqs: dict[tuple[bytes, ...], int] = Counter()
    for text in texts:
        for m in regex.finditer(PATTERN, text):
            token_tuple = tuple(bytes([b]) for b in m.group(0).encode("utf-8"))
            if len(token_tuple) > 1:
                word_freqs[token_tuple] += 1
    return _run_bpe(word_freqs, num_merges)


def reference_boundless_bpe(texts, num_merges):
    word_freqs: dict[tuple[bytes, ...], int] = Counter()
    for text in texts:
        token_tuple = tuple(bytes([b]) for b in text.encode("utf-8"))
        if len(token_tuple) > 1:
            word_freqs[token_tuple] += 1
    return _run_bpe(word_freqs, num_merges)


class TestReferenceComparison:
    def test_bpe_matches_reference_small(self):
        texts = ["the teacher teaches the thick thing"]
        ours = BPETokenizer().train(texts, num_merges=5)
        assert ours == reference_bpe(texts, num_merges=5)

    def test_bpe_matches_reference_medium(self):
        texts = ["the teacher teaches the thick thing " * 20] * 5
        ours = BPETokenizer().train(texts, num_merges=15)
        assert ours == reference_bpe(texts, num_merges=15)

    def test_fast_bpe_matches_reference(self):
        texts = ["the teacher teaches the thick thing " * 20] * 5
        fast = FastBPETrainer(texts)
        fast.train(num_merges=15)
        assert fast.get_merges() == reference_bpe(texts, num_merges=15)

    def test_boundless_bpe_matches_reference(self):
        texts = ["abcabc"]
        ours = BoundlessBPETokenizer().train(texts, num_merges=3)
        assert ours == reference_boundless_bpe(texts, num_merges=3)

    def test_all_three_bpe_match_on_same_input(self):
        texts = ["hello world hello world hello"]
        num = 5

        graph_bpe = BPETokenizer().train(texts, num_merges=num)
        fast = FastBPETrainer(texts)
        fast.train(num_merges=num)
        ref = reference_bpe(texts, num_merges=num)

        assert graph_bpe == ref
        assert fast.get_merges() == ref
