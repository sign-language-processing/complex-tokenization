from complex_tokenization.graph import Node, NodesSequence
from complex_tokenization.graphs.settings import GraphSettings
from complex_tokenization.graphs.units import utf8


class TestSingletons:
    def test_singletons_off_creates_distinct_objects(self):
        GraphSettings.USE_SINGLETONS = False
        a = Node(value=b'a')
        b = Node(value=b'a')
        assert a == b
        assert a is not b

    def test_singletons_on_returns_same_object(self):
        GraphSettings.USE_SINGLETONS = True
        a = Node(value=b'a')
        b = Node(value=b'a')
        assert a is b

    def test_singletons_different_values_different_objects(self):
        GraphSettings.USE_SINGLETONS = True
        a = Node(value=b'a')
        b = Node(value=b'b')
        assert a is not b

    def test_singletons_different_classes_not_shared(self):
        GraphSettings.USE_SINGLETONS = True
        node = Node(value=b'a')
        seq = NodesSequence(nodes=(node,))
        assert type(node) is not type(seq)

    def test_singleton_merge_preserves_identity(self):
        GraphSettings.USE_SINGLETONS = True
        graph = utf8("aa")
        assert isinstance(graph, NodesSequence)
        assert graph.nodes[0] is graph.nodes[1]
