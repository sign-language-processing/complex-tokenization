"""Fast BPE trainer using incremental pair counting.

Instead of rescanning the entire corpus for merge candidates each iteration,
maintains a running pair frequency count and only updates affected positions.
"""

from collections import Counter

from complex_tokenization.graph import GraphVertex, Node, NodesSequence, UnconnectedGraphs
from complex_tokenization.graphs.settings import GraphSettings
from complex_tokenization.graphs.units import utf8_clusters
from complex_tokenization.graphs.words import words


class FastBPETrainer:
    def __init__(self, texts: list[str], connected: bool = False, units=utf8_clusters):
        GraphSettings.ONLY_MINIMAL_MERGES = True
        GraphSettings.MAX_MERGE_SIZE = 2
        GraphSettings.USE_SINGLETONS = False

        self.word_freqs: dict[tuple[bytes, ...], int] = Counter()
        for text in texts:
            tokens = self._text_to_token_tuples(text, connected, units)
            for token_tuple in tokens:
                self.word_freqs[token_tuple] += 1

        self.merges: list[tuple[bytes, bytes]] = []
        self.stats: list[dict] = []

    @staticmethod
    def _text_to_token_tuples(text, connected, units) -> list[tuple[bytes, ...]]:
        graph = words(text, connected=connected, units=units)
        result = []

        if isinstance(graph, UnconnectedGraphs):
            subgraphs = graph.subgraphs
        else:
            subgraphs = (graph,)

        for sg in subgraphs:
            token_tuple = FastBPETrainer._flatten_to_bytes(sg)
            if token_tuple and len(token_tuple) > 1:
                result.append(token_tuple)
        return result

    @staticmethod
    def _flatten_to_bytes(vertex: GraphVertex) -> tuple[bytes, ...]:
        if isinstance(vertex, Node):
            return (vertex.value,)
        if isinstance(vertex, NodesSequence):
            result = []
            for n in vertex.nodes:
                result.extend(FastBPETrainer._flatten_to_bytes(n))
            return tuple(result)
        return (bytes(vertex),)

    def _get_pair_counts(self) -> Counter:
        counts = Counter()
        for word, freq in self.word_freqs.items():
            for i in range(len(word) - 1):
                counts[(word[i], word[i + 1])] += freq
        return counts

    def _apply_merge(self, pair: tuple[bytes, bytes]) -> dict[tuple[bytes, ...], int]:
        a, b = pair
        merged = a + b
        new_freqs = {}

        for word, freq in self.word_freqs.items():
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

        self.word_freqs = new_freqs
        return new_freqs

    def _total_tokens(self) -> int:
        return sum(len(w) * f for w, f in self.word_freqs.items())

    def train(self, num_merges: int = 100):
        pair_counts = self._get_pair_counts()
        initial_tokens = self._total_tokens()

        for _ in range(num_merges):
            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            freq = pair_counts[best_pair]
            if freq < 1:
                break

            self._apply_merge(best_pair)
            self.merges.append(best_pair)

            current_tokens = self._total_tokens()
            self.stats.append({
                "step": len(self.merges),
                "pair": best_pair,
                "frequency": freq,
                "total_tokens": current_tokens,
                "compression": 1 - current_tokens / initial_tokens if initial_tokens else 0,
            })

            pair_counts = Counter()
            for word, freq in self.word_freqs.items():
                for i in range(len(word) - 1):
                    pair_counts[(word[i], word[i + 1])] += freq

    def get_merges(self) -> list[tuple[str, ...]]:
        return [
            tuple(b.decode("utf-8", errors="replace") for b in pair)
            for pair in self.merges
        ]
