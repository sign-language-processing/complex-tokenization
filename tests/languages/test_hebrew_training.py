"""Test training a tokenizer on Hebrew text with diacritics decomposition."""

from complex_tokenization.graphs.settings import GraphSettings
from complex_tokenization.graphs.units import register_script, utf8_clusters
from complex_tokenization.graphs.words import pretokenize
from complex_tokenization.languages.hebrew.decompose import decompose_cluster
from complex_tokenization.trainer import Trainer


def train_hebrew(texts, num_merges=10):
    register_script("Hebrew", decompose_cluster)
    GraphSettings.ONLY_MINIMAL_MERGES = True
    GraphSettings.MAX_MERGE_SIZE = 2

    graphs = tuple(utf8_clusters(t) for t in texts)
    trainer = Trainer(graphs=graphs)
    trainer.train(num_merges=num_merges)
    return trainer


class TestHebrewTraining:
    def test_simple_word_training(self):
        texts = ["שלום שלום שלום"]
        trainer = train_hebrew(texts, num_merges=5)
        assert len(trainer.merges) > 0

    def test_nikkud_text_training(self):
        texts = ["בְּרֵאשִׁית בָּרָא אֱלֹהִים"] * 3
        trainer = train_hebrew(texts, num_merges=10)
        assert len(trainer.merges) > 0

    def test_repeated_diacritics_merge(self):
        """Shared diacritics across words should produce frequent merges."""
        texts = ["בָּ כָּ דָּ גָּ פָּ תָּ"] * 5
        trainer = train_hebrew(texts, num_merges=15)
        merge_bytes = [
            b"".join(bytes(n) for n in nodes)
            for _, nodes in trainer.merges
        ]
        dagesh = "ּ".encode()
        qamats = "ָ".encode()
        assert any(dagesh in mb or qamats in mb for mb in merge_bytes), (
            "Expected dagesh or qamats in early merges"
        )

    def test_mixed_nikkud_and_plain(self):
        texts = ["שלום עולם", "בְּרֵאשִׁית"]
        trainer = train_hebrew(texts, num_merges=5)
        assert len(trainer.merges) > 0

    def test_bytes_preserved(self):
        register_script("Hebrew", decompose_cluster)
        text = "שלום"
        graph = utf8_clusters(text)
        assert bytes(graph) == text.encode()

    def test_pretokenize_hebrew(self):
        text = "שלום עולם"
        tokens = pretokenize(text)
        assert len(tokens) == 2
        assert tokens[0] == "שלום"
        assert tokens[1] == " עולם"
