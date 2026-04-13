from functools import reduce

from complex_tokenization_fast._rs import Node, Trainer
from complex_tokenization_fast.graphs.settings import GraphSettings
from complex_tokenization_fast.graphs.units import characters, register_script, utf8, utf8_clusters
from complex_tokenization_fast.graphs.words import GPTPretokenizer, words

UNIT_FUNCTIONS = {
    "utf8": utf8,
    "utf8_clusters": utf8_clusters,
    "characters": characters,
}


class Tokenizer:
    def __init__(self, units="utf8_clusters", merge_size=2, connected=False, pretokenizer=GPTPretokenizer):
        if isinstance(units, str):
            if units not in UNIT_FUNCTIONS:
                raise ValueError(f"Unknown units: {units!r}. Choose from {list(UNIT_FUNCTIONS)}")
            self.units = UNIT_FUNCTIONS[units]
        else:
            self.units = units
        self.merge_size = merge_size
        self.connected = connected
        self.pretokenizer = pretokenizer
        self.merges = []

    @staticmethod
    def register_script(script, handler):
        register_script(script, handler)

    def add_merges(self, merges):
        self.merges.extend(merges)

    def _build_graphs(self, texts):
        return tuple(
            words(text, connected=self.connected, units=self.units, pretokenizer=self.pretokenizer)
            for text in texts
        )

    def make_trainer(self, texts):
        GraphSettings.ONLY_MINIMAL_MERGES = True
        GraphSettings.MAX_MERGE_SIZE = self.merge_size

        graphs = self._build_graphs(texts)
        trainer = Trainer(graphs=graphs)

        for merge_strs in self.merges:
            nodes = tuple(Node(value=s.encode("utf-8")) for s in merge_strs)
            token = reduce(lambda a, b: a + b, nodes)
            trainer.apply_merge(token, nodes)

        return trainer

    def train(self, texts, num_merges=100, progress=False):
        trainer = self.make_trainer(texts)
        _, merges = self.train_on_trainer(trainer, num_merges=num_merges, progress=progress)
        return merges

    def train_on_trainer(self, trainer, num_merges=100, progress=False):
        GraphSettings.ONLY_MINIMAL_MERGES = True
        GraphSettings.MAX_MERGE_SIZE = self.merge_size

        trainer.train(num_merges=num_merges)
        self.merges = [tuple(m) for m in trainer.get_merges()]
        return trainer, self.merges

    def get_merges(self):
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
    def __init__(self, units="utf8_clusters", disconnected_merges=None, pretokenizer=GPTPretokenizer):
        super().__init__(units=units, merge_size=2, connected=False, pretokenizer=pretokenizer)
        self._disconnected_merges = disconnected_merges

    def train(self, texts, num_merges=100, progress=False):
        disconnected_merges = self._disconnected_merges or num_merges // 2
        phase1 = BPETokenizer(units=self.units, pretokenizer=self.pretokenizer)
        phase1.train(texts, num_merges=disconnected_merges)
        self.connected = True
        self.add_merges(phase1.merges)
        return super().train(texts, num_merges=num_merges)
