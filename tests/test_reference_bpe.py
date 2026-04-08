"""Compare our BPE implementations against a simple reference implementation.

The reference implementation is a straightforward BPE that operates on
byte sequences, without any graph abstraction. It serves as ground truth.
"""

from collections import Counter

import regex

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
        from complex_tokenization.examples.bpe import train_bpe_tokenizer
        texts = ["the teacher teaches the thick thing"]
        ours = train_bpe_tokenizer(texts, num_merges=5)
        ref = reference_bpe(texts, num_merges=5)
        assert ours == ref

    def test_bpe_matches_reference_medium(self):
        from complex_tokenization.examples.bpe import train_bpe_tokenizer
        texts = ["the teacher teaches the thick thing " * 20] * 5
        ours = train_bpe_tokenizer(texts, num_merges=15)
        ref = reference_bpe(texts, num_merges=15)
        assert ours == ref

    def test_fast_bpe_matches_reference(self):
        from complex_tokenization.fast_bpe_trainer import FastBPETrainer
        texts = ["the teacher teaches the thick thing " * 20] * 5
        fast = FastBPETrainer(texts)
        fast.train(num_merges=15)
        ref = reference_bpe(texts, num_merges=15)
        assert fast.get_merges() == ref

    def test_boundless_bpe_matches_reference(self):
        from complex_tokenization.examples.boundless_bpe import train_boundless_bpe_tokenizer
        texts = ["abcabc"]
        ours = train_boundless_bpe_tokenizer(texts, num_merges=3)
        ref = reference_boundless_bpe(texts, num_merges=3)
        assert ours == ref

    def test_all_three_bpe_match_on_same_input(self):
        from complex_tokenization.examples.bpe import train_bpe_tokenizer
        from complex_tokenization.fast_bpe_trainer import FastBPETrainer
        texts = ["hello world hello world hello"]
        num = 5

        graph_bpe = train_bpe_tokenizer(texts, num_merges=num)
        fast = FastBPETrainer(texts)
        fast.train(num_merges=num)
        ref = reference_bpe(texts, num_merges=num)

        assert graph_bpe == ref
        assert fast.get_merges() == ref
