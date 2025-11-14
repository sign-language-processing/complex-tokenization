from collections import Counter

from complex_tokenization.graph import GraphVertex, NodesSequence, Node
from complex_tokenization.graphs.settings import GraphSettings
from complex_tokenization.graphs.utf8 import utf8, utf8_clusters
from complex_tokenization.graphs.words import words


def readable_merges(graph: GraphVertex):
    counter = Counter(graph.get_merges())
    byte_merges = {}
    for nodes, v in counter.items():
        k = b''.join(bytes(node) for node in nodes)
        byte_merges[k] = v
    return byte_merges


class TestUTF8Word:
    def test_utf8_ascii_same_as_cluster(self):
        assert utf8_clusters('word') == utf8('word')

    def test_utf8_cluster_is_split(self):
        graph = utf8_clusters('שלום')
        assert isinstance(graph, NodesSequence)
        assert len(graph.nodes) == 4
        for node in graph.nodes:
            assert isinstance(node, NodesSequence)
            assert len(node.nodes) == 2
            assert isinstance(node.nodes[0], Node)

    def test_utf8_ascii_2_merges(self):
        GraphSettings.MAX_MERGE_SIZE = 2

        graph = utf8('lalaland')
        merges = readable_merges(graph)
        assert len(merges) == 4

        assert merges[b'la'] == 3
        assert merges[b'al'] == 2
        assert merges[b'an'] == 1
        assert merges[b'nd'] == 1

    def test_utf8_ascii_3_merges(self):
        GraphSettings.MAX_MERGE_SIZE = 3

        graph = utf8('lalaland')
        merges = readable_merges(graph)
        assert len(merges) == 8

        assert merges[b'la'] == 3
        assert merges[b'lal'] == 2
        assert merges[b'al'] == 2
        assert merges[b'ala'] == 2
        assert merges[b'lan'] == 1
        assert merges[b'an'] == 1
        assert merges[b'and'] == 1
        assert merges[b'nd'] == 1

    def test_utf8_cluster_minimal_merges(self):
        GraphSettings.MAX_MERGE_SIZE = 100
        GraphSettings.ONLY_MINIMAL_MERGES = True
        graph = utf8_clusters('שלום')
        merges = readable_merges(graph)

        print(merges)
        # Only character sequences should be valid
        assert merges['ש'.encode('utf-8')] == 1
        assert merges['ל'.encode('utf-8')] == 1
        assert merges['ו'.encode('utf-8')] == 1
        assert merges['ם'.encode('utf-8')] == 1

    def test_utf8_cluster_non_minimal_merges(self):
        GraphSettings.MAX_MERGE_SIZE = 100
        GraphSettings.ONLY_MINIMAL_MERGES = False
        graph = utf8_clusters('שלום')
        merges = readable_merges(graph)

        # Basically, every subsequence is valid
        assert merges['ש'.encode('utf-8')] == 1
        assert merges['ל'.encode('utf-8')] == 1
        assert merges['ו'.encode('utf-8')] == 1
        assert merges['ם'.encode('utf-8')] == 1
        assert merges['של'.encode('utf-8')] == 1
        assert merges['שלו'.encode('utf-8')] == 1
        assert merges['שלום'.encode('utf-8')] == 1
        assert merges['לו'.encode('utf-8')] == 1
        assert merges['לום'.encode('utf-8')] == 1
        assert merges['ום'.encode('utf-8')] == 1


class TestWords:
    def test_single_word_same_as_utf8_clusters(self):
        # Single word should be identical to utf8_clusters
        assert words('word') == utf8_clusters('word')
        assert words('שלום') == utf8_clusters('שלום')

    def test_multiple_words_count(self):
        # Test that multiple words are properly split and counted
        graph = words('hello world test')
        assert isinstance(graph, NodesSequence)
        assert len(graph.nodes) == 3

    def test_two_words_minimal_merges(self):
        GraphSettings.MAX_MERGE_SIZE = 10
        GraphSettings.ONLY_MINIMAL_MERGES = True

        graph = words('hi bye')
        merges = readable_merges(graph)
        assert len(merges) == 7

        assert merges[b'hi'] == 1
        assert merges[b' b'] == 1
        assert merges[b' by'] == 1
        assert merges[b' bye'] == 1
        assert merges[b'by'] == 1
        assert merges[b'bye'] == 1
        assert merges[b'ye'] == 1

    def test_two_words_non_minimal_merge(self):
        GraphSettings.MAX_MERGE_SIZE = 10
        GraphSettings.ONLY_MINIMAL_MERGES = False

        graph = words('hi bye')
        merges = readable_merges(graph)
        assert len(merges) == 8

        assert merges[b'hi bye'] == 1
