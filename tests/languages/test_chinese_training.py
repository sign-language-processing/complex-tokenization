"""Test training a tokenizer on Chinese text with IDS decomposition."""

from complex_tokenization.graph import Node, NodesSequence, Tree
from complex_tokenization.graphs.settings import GraphSettings
from complex_tokenization.graphs.units import register_script, utf8_clusters
from complex_tokenization.languages.chinese.graph import chinese_character_to_graph
from complex_tokenization.trainer import Trainer


class TestChineseGraph:
    def test_decomposable_character(self):
        graph = chinese_character_to_graph("林")
        assert isinstance(graph, Tree)
        assert bytes(graph.root) == "⿰".encode()

    def test_non_decomposable_character(self):
        graph = chinese_character_to_graph("a")
        assert isinstance(graph, Node)

    def test_chinese_text_via_registry(self):
        register_script("Han", chinese_character_to_graph)
        graph = utf8_clusters("林木")
        assert isinstance(graph, NodesSequence)
        assert bytes(graph) == "⿰木木木".encode()

    def test_mixed_text_via_registry(self):
        register_script("Han", chinese_character_to_graph)
        graph = utf8_clusters("hello")
        assert bytes(graph) == b"hello"


class TestChineseTraining:
    def test_train_on_repeated_characters(self):
        register_script("Han", chinese_character_to_graph)
        GraphSettings.ONLY_MINIMAL_MERGES = True
        GraphSettings.MAX_MERGE_SIZE = 2

        texts = ["林森木本末朱机杏"] * 3
        graphs = tuple(utf8_clusters(t) for t in texts)
        trainer = Trainer(graphs=graphs)
        trainer.train(num_merges=5)

        assert len(trainer.get_merges()) > 0

    def test_train_on_mixed_chinese_text(self):
        register_script("Han", chinese_character_to_graph)
        GraphSettings.ONLY_MINIMAL_MERGES = True
        GraphSettings.MAX_MERGE_SIZE = 2

        texts = ["你好世界 hello 你好"]
        graphs = tuple(utf8_clusters(t) for t in texts)
        trainer = Trainer(graphs=graphs)
        trainer.train(num_merges=3)
        assert len(trainer.merges) <= 3

    def test_common_radicals_merge_early(self):
        register_script("Han", chinese_character_to_graph)
        GraphSettings.ONLY_MINIMAL_MERGES = True
        GraphSettings.MAX_MERGE_SIZE = 2

        texts = ["林森林森林森"] * 5
        graphs = tuple(utf8_clusters(t) for t in texts)
        trainer = Trainer(graphs=graphs)
        trainer.train(num_merges=20)

        merge_bytes = [
            b"".join(bytes(n) for n in nodes)
            for _, nodes in trainer.merges
        ]
        wood = "木".encode()
        assert any(wood in mb for mb in merge_bytes), (
            "Expected '木' in merge bytes within 20 merges"
        )
