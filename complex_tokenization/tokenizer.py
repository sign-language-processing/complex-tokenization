"""High-level tokenizer API.

Usage:
    tokenizer = BPETokenizer()
    tokenizer.train(texts, num_merges=100)
    merges = tokenizer.get_merges()

With language-specific decomposition:
    from complex_tokenization.languages.hebrew.decompose import decompose_cluster
    tokenizer = BPETokenizer()
    tokenizer.register_script("Hebrew", decompose_cluster)
    tokenizer.train(texts, num_merges=100)
"""

from collections.abc import Callable
from functools import reduce

from complex_tokenization.graph import GraphVertex, Node
from complex_tokenization.graphs.settings import GraphSettings
from complex_tokenization.graphs.units import characters, register_script, utf8, utf8_clusters
from complex_tokenization.graphs.words import words
from complex_tokenization.trainer import Trainer

UNIT_FUNCTIONS: dict[str, Callable[[str], GraphVertex]] = {
    "utf8": utf8,
    "utf8_clusters": utf8_clusters,
    "characters": characters,
}


class Tokenizer:
    def __init__(
        self,
        units: str | Callable[[str], GraphVertex] = "utf8_clusters",
        merge_size: int = 2,
        connected: bool = False,
    ):
        if isinstance(units, str):
            if units not in UNIT_FUNCTIONS:
                raise ValueError(f"Unknown units: {units!r}. Choose from {list(UNIT_FUNCTIONS)}")
            self.units = UNIT_FUNCTIONS[units]
        else:
            self.units = units
        self.merge_size = merge_size
        self.connected = connected
        self.trainer: Trainer | None = None

    @staticmethod
    def register_script(script: str, handler: Callable[[str], GraphVertex]):
        register_script(script, handler)

    def _build_graphs(self, texts: list[str]) -> tuple[GraphVertex, ...]:
        return tuple(
            words(text, connected=self.connected, units=self.units)
            for text in texts
        )

    def train(self, texts: list[str], num_merges: int = 100,
              known_merges: list[tuple[str, ...]] | None = None) -> list:
        GraphSettings.ONLY_MINIMAL_MERGES = True
        GraphSettings.MAX_MERGE_SIZE = self.merge_size

        graphs = self._build_graphs(texts)
        self.trainer = Trainer(graphs=graphs)

        if known_merges:
            for merge_strs in known_merges:
                nodes = tuple(Node(value=s.encode("utf-8")) for s in merge_strs)
                token = reduce(lambda a, b: a + b, nodes)
                self.trainer.graph = self.trainer.graph.merge(token, nodes)
                self.trainer.merges.append((token, nodes))

        self.trainer.train(num_merges=num_merges)
        return self.get_merges()

    def get_merges(self) -> list[tuple[str, ...]]:
        if self.trainer is None:
            return []
        return self.trainer.get_merges()


class BPETokenizer(Tokenizer):
    def __init__(self, units="utf8_clusters"):
        super().__init__(units=units, merge_size=2, connected=False)


class BNETokenizer(Tokenizer):
    def __init__(self, n=4, units="utf8_clusters"):
        super().__init__(units=units, merge_size=n, connected=False)


class BoundlessBPETokenizer(Tokenizer):
    def __init__(self, units="utf8_clusters"):
        super().__init__(units=units, merge_size=2, connected=True)


class SuperBPETokenizer(Tokenizer):
    def __init__(self, units="utf8_clusters", disconnected_merges: int | None = None):
        super().__init__(units=units, merge_size=2, connected=False)
        self._disconnected_merges = disconnected_merges

    def train(self, texts: list[str], num_merges: int = 100,
              known_merges: list[tuple[str, ...]] | None = None) -> list:
        disconnected_merges = self._disconnected_merges or num_merges // 2

        phase1 = BPETokenizer(units=self.units)
        phase1_merges = phase1.train(texts, num_merges=disconnected_merges, known_merges=known_merges)

        phase2 = BoundlessBPETokenizer(units=self.units)
        result = phase2.train(texts, num_merges=num_merges, known_merges=phase1_merges)

        self.trainer = phase2.trainer
        return result
