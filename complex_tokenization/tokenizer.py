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

from tokenizers.pre_tokenizers import PreTokenizer

from complex_tokenization.graph import GraphVertex, Node
from complex_tokenization.graphs.settings import GraphSettings
from complex_tokenization.graphs.units import characters, register_script, utf8, utf8_clusters
from complex_tokenization.graphs.words import GPTPretokenizer, words
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
        pretokenizer: PreTokenizer = GPTPretokenizer,
    ):
        if isinstance(units, str):
            if units not in UNIT_FUNCTIONS:
                raise ValueError(f"Unknown units: {units!r}. Choose from {list(UNIT_FUNCTIONS)}")
            self.units = UNIT_FUNCTIONS[units]
        else:
            self.units = units
        self.merge_size = merge_size
        self.connected = connected
        self.pretokenizer = pretokenizer
        self.merges: list[tuple[str, ...]] = []

    @staticmethod
    def register_script(script: str, handler: Callable[[str], GraphVertex]):
        register_script(script, handler)

    def add_merges(self, merges: list[tuple[str, ...]]):
        self.merges.extend(merges)

    def _build_graphs(self, texts: list[str]) -> tuple[GraphVertex, ...]:
        return tuple(
            words(text, connected=self.connected, units=self.units,
                  pretokenizer=self.pretokenizer)
            for text in texts
        )

    def train(self, texts: list[str], num_merges: int = 100, progress: bool = False) -> list[tuple[str, ...]]:
        GraphSettings.ONLY_MINIMAL_MERGES = True
        GraphSettings.MAX_MERGE_SIZE = self.merge_size

        graphs = self._build_graphs(texts)
        trainer = Trainer(graphs=graphs)

        for merge_strs in self.merges:
            nodes = tuple(Node(value=s.encode("utf-8")) for s in merge_strs)
            token = reduce(lambda a, b: a + b, nodes)
            trainer.graph = trainer.graph.merge(token, nodes)
            trainer.merges.append((token, nodes))

        trainer.train(num_merges=num_merges, progress=progress)
        self.merges = trainer.get_merges()
        return self.merges

    def get_merges(self) -> list[tuple[str, ...]]:
        return list(self.merges)


class BPETokenizer(Tokenizer):
    def __init__(self, units="utf8_clusters", pretokenizer=GPTPretokenizer):
        super().__init__(units=units, merge_size=2, connected=False, pretokenizer=pretokenizer)


class BNETokenizer(Tokenizer):
    def __init__(self, n=4, units="utf8_clusters", pretokenizer=GPTPretokenizer):
        super().__init__(units=units, merge_size=n, connected=False, pretokenizer=pretokenizer)


class BoundlessBPETokenizer(Tokenizer):
    def __init__(self, units="utf8_clusters", pretokenizer=GPTPretokenizer):
        super().__init__(units=units, merge_size=2, connected=True, pretokenizer=pretokenizer)


class SuperBPETokenizer(Tokenizer):
    def __init__(self, units="utf8_clusters", disconnected_merges: int | None = None, pretokenizer=GPTPretokenizer):
        super().__init__(units=units, merge_size=2, connected=False, pretokenizer=pretokenizer)
        self._disconnected_merges = disconnected_merges

    def train(self, texts: list[str], num_merges: int = 100, progress: bool = False) -> list[tuple[str, ...]]:
        disconnected_merges = self._disconnected_merges or num_merges // 2

        phase1 = BPETokenizer(units=self.units, pretokenizer=self.pretokenizer)
        phase1.train(texts, num_merges=disconnected_merges, progress=progress)

        self.connected = True
        self.add_merges(phase1.merges)
        return super().train(texts, num_merges=num_merges, progress=progress)
