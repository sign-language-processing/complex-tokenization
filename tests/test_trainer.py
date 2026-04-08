import pytest

from complex_tokenization.graph import Node, NodesSequence
from complex_tokenization.graphs.settings import GraphSettings
from complex_tokenization.graphs.units import utf8
from complex_tokenization.trainer import Trainer


class TestTrainer:
    def test_trainer_requires_graph_or_graphs(self):
        with pytest.raises(ValueError, match="Must provide either graph or graphs"):
            Trainer()

    def test_trainer_rejects_both_graph_and_graphs(self):
        graph = utf8("test")
        with pytest.raises(ValueError, match="Must provide either graph or graphs, not both"):
            Trainer(graph=graph, graphs=(graph,))

    def test_train_single_node_no_merges(self):
        GraphSettings.MAX_MERGE_SIZE = 2
        GraphSettings.ONLY_MINIMAL_MERGES = True
        node = Node(value=b'a')
        trainer = Trainer(graph=node)
        trainer.train(num_merges=10)
        assert len(trainer.merges) == 0

    def test_train_stops_when_no_merges_left(self):
        GraphSettings.MAX_MERGE_SIZE = 2
        GraphSettings.ONLY_MINIMAL_MERGES = True
        graph = utf8("ab")
        trainer = Trainer(graph=graph)
        trainer.train(num_merges=100)
        assert len(trainer.merges) == 1

    def test_train_merge_reduces_graph(self):
        GraphSettings.MAX_MERGE_SIZE = 2
        GraphSettings.ONLY_MINIMAL_MERGES = True
        graph = utf8("aaa")
        trainer = Trainer(graph=graph)
        trainer.train(num_merges=1)
        assert len(trainer.merges) == 1
        assert isinstance(trainer.graph, NodesSequence)

    def test_train_full_merge_to_single_node(self):
        GraphSettings.MAX_MERGE_SIZE = 2
        GraphSettings.ONLY_MINIMAL_MERGES = True
        graph = utf8("aa")
        trainer = Trainer(graph=graph)
        trainer.train(num_merges=1)
        assert len(trainer.merges) == 1
        assert isinstance(trainer.graph, Node)

    def test_get_merges_returns_readable(self):
        GraphSettings.MAX_MERGE_SIZE = 2
        GraphSettings.ONLY_MINIMAL_MERGES = True
        graph = utf8("abab")
        trainer = Trainer(graph=graph)
        trainer.train(num_merges=1)
        merges = trainer.get_merges()
        assert len(merges) == 1
        assert merges[0] == ('a', 'b')

    def test_train_with_multiple_graphs(self):
        GraphSettings.MAX_MERGE_SIZE = 2
        GraphSettings.ONLY_MINIMAL_MERGES = True
        graphs = (utf8("ab"), utf8("ab"), utf8("cd"))
        trainer = Trainer(graphs=graphs)
        trainer.train(num_merges=1)
        assert trainer.get_merges()[0] == ('a', 'b')

    def test_characters_produce_valid_bytes(self):
        from complex_tokenization.graphs.units import characters
        graph = characters("hello")
        assert bytes(graph) == b"hello"

    def test_characters_non_ascii_produce_valid_bytes(self):
        from complex_tokenization.graphs.units import characters
        graph = characters("שלום")
        assert bytes(graph) == "שלום".encode()
