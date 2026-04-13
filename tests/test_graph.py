from collections import Counter

from complex_tokenization.graph import GraphVertex, Node, NodesSequence
from complex_tokenization.graphs.settings import GraphSettings
from complex_tokenization.graphs.units import characters, utf8, utf8_clusters
from complex_tokenization.graphs.words import words


def readable_merges(graph: GraphVertex):
    counter = Counter(graph.get_merges())
    byte_merges = {}
    for nodes, v in counter.items():
        k = b''.join(bytes(node) for node in nodes)
        byte_merges[k] = v
    return byte_merges


class TestNodeCount:
    def test_single_node(self):
        assert Node(b'a').node_count() == 1

    def test_utf8_bytes(self):
        graph = utf8("abc")
        assert graph.node_count() == 3

    def test_utf8_clusters(self):
        graph = utf8_clusters("שלום")
        assert graph.node_count() == 8  # 4 chars × 2 bytes each

    def test_words(self):
        graph = words("hi bye")
        assert graph.node_count() == 6  # h, i, ' ', b, y, e

    def test_after_merge(self):
        graph = utf8("aabb")
        assert graph.node_count() == 4
        merged = graph.merge(Node(b'aa'), (Node(b'a'), Node(b'a')))
        assert merged.node_count() == 3  # aa, b, b

    def test_node_count_matches_train_with_counts(self):
        """Ensure train_with_counts reports the same node_count as graph.node_count()."""
        from complex_tokenization.tokenizer import BPETokenizer

        tok = BPETokenizer()
        texts = ["the teacher teaches the thick thing"] * 3
        trainer = tok.make_trainer(texts)
        initial = trainer.graph.node_count()

        if hasattr(trainer, 'train_with_counts'):
            xs, ys = trainer.train_with_counts(5, 1)
            assert ys[0] == initial, f"Initial count mismatch: {ys[0]} != {initial}"

            # Also check after training via the normal path
            tok2 = BPETokenizer()
            trainer2 = tok2.make_trainer(texts)
            trainer2, _ = tok2.train_on_trainer(trainer2, num_merges=5)
            after_normal = trainer2.graph.node_count()
            assert ys[5] == after_normal, f"After 5 merges mismatch: {ys[5]} != {after_normal}"


    def test_node_count_with_chinese_ids(self):
        """Ensure node_count is correct when Chinese IDS decomposition is used."""
        from complex_tokenization.tokenizer import BPETokenizer
        from complex_tokenization.graphs.units import register_script
        from complex_tokenization.languages.chinese.graph import chinese_character_to_graph

        register_script("Han", chinese_character_to_graph)
        tok = BPETokenizer()
        texts = ["林森木本末"] * 3
        trainer = tok.make_trainer(texts)
        initial = trainer.graph.node_count()
        assert initial > 0

        if hasattr(trainer, 'train_with_counts'):
            xs, ys = trainer.train_with_counts(5, 1)
            assert ys[0] == initial

            tok2 = BPETokenizer()
            trainer2 = tok2.make_trainer(texts)
            for i in range(1, 6):
                trainer2, _ = tok2.train_on_trainer(trainer2, num_merges=i)
                assert ys[i] == trainer2.graph.node_count(), \
                    f"Merge {i}: train_with_counts={ys[i]} vs train_on_trainer={trainer2.graph.node_count()}"

    def test_collapsed_trees_merge_as_pairs(self):
        """After tree merges collapse Trees to Nodes in a MixedSeq, they should become
        eligible for pair merges at the parent Seq level."""
        from complex_tokenization.graph import Node, NodesSequence, Tree, UnconnectedGraphs
        from complex_tokenization.graphs.settings import GraphSettings
        GraphSettings.MAX_MERGE_SIZE = 2
        GraphSettings.ONLY_MINIMAL_MERGES = True

        c1 = Tree(root=NodesSequence((Node(b'\xe2'), Node(b'\xbf'), Node(b'\xb1'))),
                  children=(NodesSequence((Node(b'\xe4'), Node(b'\xb8'), Node(b'\xb6'))),
                            NodesSequence((Node(b'\xe4'), Node(b'\xb8'), Node(b'\x80')))))
        c2 = Tree(root=NodesSequence((Node(b'\xe2'), Node(b'\xbf'), Node(b'\xb0'))),
                  children=(NodesSequence((Node(b'\xe6'), Node(b'\x9c'), Node(b'\xa8'))),
                            NodesSequence((Node(b'\xe6'), Node(b'\x9c'), Node(b'\xa8')))))
        word = NodesSequence((c1, c2))
        ug = UnconnectedGraphs((word,) * 100)

        # Use the Tokenizer API to ensure settings are synced
        from complex_tokenization.tokenizer import BPETokenizer
        tok = BPETokenizer()
        tok.merges = []
        from complex_tokenization.trainer import Trainer
        trainer = Trainer(graph=ug)
        trainer.train(num_merges=30)
        merges = trainer.get_merges()
        # After byte merges + tree merges + final pair merge: 100 nodes (one per word)
        assert len(merges) >= 11, f"Expected at least 11 merges, got {len(merges)}"
        assert trainer.graph.node_count() == 100, \
            f"Expected 100 nodes, got {trainer.graph.node_count()}"

    def test_tree_merges_fire_with_chinese_ids(self):
        """After enough byte merges, tree merges should fire for Chinese IDS."""
        from complex_tokenization.tokenizer import BPETokenizer
        from complex_tokenization.graphs.units import register_script
        from complex_tokenization.languages.chinese.graph import chinese_character_to_graph

        register_script("Han", chinese_character_to_graph)
        tok = BPETokenizer()
        # 亮 = ⿱(丶, 一) — after byte merges reduce ⿱/丶/一 to single Nodes, tree merge fires
        texts = ["亮亮亮亮亮亮亮亮亮亮"] * 5
        merges = tok.train(texts, num_merges=20)
        # At least one merge should be a tree merge (len > 2)
        tree_merges = [m for m in merges if len(m) > 2]
        assert len(tree_merges) > 0, f"Expected tree merges in {merges}"


class TestUnitsWord:
    def test_characters_split(self):
        assert characters("שלום") == NodesSequence((
            Node("ש".encode()), Node("ל".encode()), Node("ו".encode()), Node("ם".encode())))

    def test_utf8_split(self):
        assert utf8("שלום") == NodesSequence((Node(value=b'\xd7'), Node(value=b'\xa9'),
                                              Node(value=b'\xd7'), Node(value=b'\x9c'),
                                              Node(value=b'\xd7'), Node(value=b'\x95'),
                                              Node(value=b'\xd7'), Node(value=b'\x9d')))

    def test_utf8_clusters_split(self):
        assert utf8_clusters("שלום") == NodesSequence((
            NodesSequence((Node(value=b'\xd7'), Node(value=b'\xa9'))),
            NodesSequence((Node(value=b'\xd7'), Node(value=b'\x9c'))),
            NodesSequence((Node(value=b'\xd7'), Node(value=b'\x95'))),
            NodesSequence((Node(value=b'\xd7'), Node(value=b'\x9d')))))

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
        assert merges['ש'.encode()] == 1
        assert merges['ל'.encode()] == 1
        assert merges['ו'.encode()] == 1
        assert merges['ם'.encode()] == 1

    def test_utf8_cluster_non_minimal_merges(self):
        GraphSettings.MAX_MERGE_SIZE = 100
        GraphSettings.ONLY_MINIMAL_MERGES = False
        graph = utf8_clusters('שלום')
        merges = readable_merges(graph)

        # Basically, every subsequence is valid
        assert merges['ש'.encode()] == 1
        assert merges['ל'.encode()] == 1
        assert merges['ו'.encode()] == 1
        assert merges['ם'.encode()] == 1
        assert merges['של'.encode()] == 1
        assert merges['שלו'.encode()] == 1
        assert merges['שלום'.encode()] == 1
        assert merges['לו'.encode()] == 1
        assert merges['לום'.encode()] == 1
        assert merges['ום'.encode()] == 1


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
