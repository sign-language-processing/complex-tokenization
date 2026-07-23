from collections import Counter

from complex_tokenization.tokenizer import BNETokenizer, BPETokenizer

# A deliberately tie-heavy corpus: at the first BPE merge, seven candidate pairs
# share the top score, so the winner is decided purely by the tie-break rule
# (first candidate in traversal/emit order, matching the reference trainer and
# HuggingFace). Expected lists are generated from the reference implementation
# and pin the contract for both implementations (fast aliases this module).
CORPUS = ["ab cd ab cd ef gh ef gh"] * 3


class TestTieBreak:
    def test_corpus_actually_has_score_ties(self):
        tok = BPETokenizer()
        trainer = tok.make_trainer(CORPUS)
        counts = Counter(trainer.graph.get_merges())
        best = max((len(m) - 1) * c for m, c in counts.items())
        tied = [m for m, c in counts.items() if (len(m) - 1) * c == best]
        assert len(tied) > 1, f"corpus must produce a score tie, got {tied}"

    def test_bpe_tie_break(self):
        merges = BPETokenizer().train(CORPUS, num_merges=10)
        assert merges == [
            ("a", "b"),
            (" ", "c"),
            (" c", "d"),
            (" ", "e"),
            (" e", "f"),
            (" ", "g"),
            (" g", "h"),
            (" ", "ab"),
        ]

    def test_bne_tie_break(self):
        merges = BNETokenizer(n=4).train(CORPUS, num_merges=10)
        assert merges == [
            (" ", "c", "d"),
            (" ", "e", "f"),
            (" ", "g", "h"),
            ("a", "b"),
            (" ", "ab"),
        ]
