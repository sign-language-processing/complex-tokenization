from functools import lru_cache, reduce

from complex_tokenization_fast._rs import (
    Node,
    Trainer,
    has_cluster_handlers_py,
    str_to_bytes,
    trainer_from_texts,
)
from complex_tokenization_fast.graphs.settings import GraphSettings
from complex_tokenization_fast.graphs.units import characters, register_script, utf8, utf8_clusters
from complex_tokenization_fast.graphs.words import GPTPretokenizer, words

UNIT_FUNCTIONS = {
    "utf8": utf8,
    "utf8_clusters": utf8_clusters,
    "characters": characters,
}


class Tokenizer:
    def __init__(self, units="utf8_clusters", merge_size=2, connected=False, pretokenizer=GPTPretokenizer,
                 cache_maxsize=None):
        if isinstance(units, str):
            if units not in UNIT_FUNCTIONS:
                raise ValueError(f"Unknown units: {units!r}. Choose from {list(UNIT_FUNCTIONS)}")
            self.units = UNIT_FUNCTIONS[units]
        else:
            self.units = units
        self.merge_size = merge_size
        self.connected = connected
        self.pretokenizer = pretokenizer
        self.cache_maxsize = cache_maxsize
        self.merges = []

    @staticmethod
    def register_script(script, handler):
        register_script(script, handler)

    def add_merges(self, merges):
        self.merges.extend(merges)

    def _build_graphs(self, texts):
        # Same build-local word-graph dedup as the reference: repeated words
        # share one graph. cache_maxsize=None is unbounded; 0 disables.
        if self.cache_maxsize == 0:
            units = self.units
        else:
            units = lru_cache(maxsize=self.cache_maxsize)(self.units)
        return tuple(
            words(text, connected=self.connected, units=units, pretokenizer=self.pretokenizer)
            for text in texts
        )

    def make_trainer(self, texts):
        GraphSettings.ONLY_MINIMAL_MERGES = True
        GraphSettings.MAX_MERGE_SIZE = self.merge_size

        # Default configuration ingests entirely in Rust (one boundary
        # crossing; same `tokenizers` crate as GPTPretokenizer, so splits are
        # identical). Custom pretokenizers/units/script handlers take the
        # per-document Python path.
        if (
            self.units is utf8_clusters
            and self.pretokenizer is GPTPretokenizer
            and not has_cluster_handlers_py()
        ):
            trainer = trainer_from_texts(list(texts), connected=self.connected)
        else:
            graphs = self._build_graphs(texts)
            trainer = Trainer(graphs=graphs)

        if self.merges:
            merge_list = []
            for merge_strs in self.merges:
                nodes = tuple(Node(value=str_to_bytes(s)) for s in merge_strs)
                token = reduce(lambda a, b: a + b, nodes)
                merge_list.append((token, nodes))
            trainer.apply_merges(merge_list)

        return trainer

    def make_streaming_trainer(self, texts):
        GraphSettings.ONLY_MINIMAL_MERGES = True
        GraphSettings.MAX_MERGE_SIZE = self.merge_size

        from complex_tokenization_fast._rs import clear_word_cache, warm_word_cache_py
        from complex_tokenization_fast.graphs.words import pretokenize

        clear_word_cache()
        doc_words = [pretokenize(text, self.pretokenizer) for text in texts]
        all_unique_words = list({w for words in doc_words for w in words})
        warm_word_cache_py(all_unique_words)

        trainer = Trainer(graph=Node(value=b""))
        trainer.set_streaming(doc_words, connected=self.connected)

        if self.merges:
            merge_list = []
            for merge_strs in self.merges:
                nodes = tuple(Node(value=str_to_bytes(s)) for s in merge_strs)
                token = reduce(lambda a, b: a + b, nodes)
                merge_list.append((token, nodes))
            trainer.apply_merges(merge_list)

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
    def __init__(self, **kwargs):
        super().__init__(merge_size=2, connected=False, **kwargs)


class BNETokenizer(Tokenizer):
    def __init__(self, n=4, **kwargs):
        super().__init__(merge_size=n, connected=False, **kwargs)


class BoundlessBPETokenizer(Tokenizer):
    def __init__(self, **kwargs):
        super().__init__(merge_size=2, connected=True, **kwargs)


class SuperBPETokenizer(Tokenizer):
    def __init__(self, disconnected_merges=None, **kwargs):
        super().__init__(merge_size=2, connected=False, **kwargs)
        self._disconnected_merges = disconnected_merges

    def train(self, texts, num_merges=100, progress=False):
        disconnected_merges = self._disconnected_merges or num_merges // 2
        phase1 = BPETokenizer(units=self.units, pretokenizer=self.pretokenizer, cache_maxsize=self.cache_maxsize)
        phase1.train(texts, num_merges=disconnected_merges)
        self.connected = True
        self.add_merges(phase1.merges)
        return super().train(texts, num_merges=num_merges)
