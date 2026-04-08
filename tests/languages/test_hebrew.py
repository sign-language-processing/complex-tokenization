import unicodedata

from complex_tokenization.graph import FullyConnectedGraph, Node, NodesSequence
from complex_tokenization.graphs.units import register_script, utf8_clusters
from complex_tokenization.languages.hebrew.decompose import decompose_cluster


class TestDecomposeCluster:
    def test_bare_letter(self):
        result = decompose_cluster("א")
        assert isinstance(result, (NodesSequence, Node))
        assert bytes(result) == "א".encode()

    def test_letter_with_one_mark(self):
        cluster = "בָ"  # bet + qamats
        result = decompose_cluster(cluster)
        assert isinstance(result, NodesSequence)
        assert bytes(result) == cluster.encode()

    def test_letter_with_dagesh_and_vowel(self):
        cluster = "בְּ"  # bet + sheva + dagesh
        result = decompose_cluster(cluster)
        assert isinstance(result, NodesSequence)
        assert bytes(result) == cluster.encode()
        base, diacritics = result.nodes
        assert isinstance(diacritics, FullyConnectedGraph)
        assert len(diacritics.nodes) == 2

    def test_letter_with_three_marks(self):
        cluster = "שִׁ֖"  # shin + hiriq + shin dot + tipeha
        result = decompose_cluster(cluster)
        assert isinstance(result, NodesSequence)
        assert bytes(result) == cluster.encode()
        base, diacritics = result.nodes
        assert isinstance(diacritics, FullyConnectedGraph)
        assert len(diacritics.nodes) == 3

    def test_single_mark_collapses(self):
        cluster = "בָ"  # bet + qamats (one mark)
        result = decompose_cluster(cluster)
        base, mark = result.nodes
        assert not isinstance(mark, FullyConnectedGraph)

    def test_fully_connected_merges(self):
        cluster = "בְּ"  # bet + sheva + dagesh
        result = decompose_cluster(cluster)
        merges = list(result.get_merges())
        merge_bytes = [b"".join(bytes(n) for n in m) for m in merges]
        sheva = "ְ".encode()
        dagesh = "ּ".encode()
        assert sheva + dagesh in merge_bytes
        assert dagesh + sheva in merge_bytes


class TestHebrewViaRegistry:
    def test_bytes_roundtrip(self):
        register_script("Hebrew", decompose_cluster)
        word = "בְּרֵאשִׁ֖ית"
        graph = utf8_clusters(word)
        assert bytes(graph) == word.encode()

    def test_simple_word(self):
        register_script("Hebrew", decompose_cluster)
        result = utf8_clusters("שלום")
        assert isinstance(result, NodesSequence)
        assert bytes(result) == "שלום".encode()

    def test_word_with_nikkud(self):
        register_script("Hebrew", decompose_cluster)
        word = "שָׁלוֹם"
        result = utf8_clusters(word)
        assert isinstance(result, NodesSequence)
        assert bytes(result) == word.encode()

    def test_mark_categories(self):
        marks = "ְִֵּׁ֖"
        for ch in marks:
            cat = unicodedata.category(ch)
            assert cat == "Mn", f"{ch!r} (U+{ord(ch):04X}) is {cat}, not Mn"
