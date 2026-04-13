"""Decompose Hebrew grapheme clusters into graph structure.

Each grapheme cluster becomes:
- A NodesSequence of [base_letter, diacritics_graph]
- Where diacritics_graph is a FullyConnectedGraph of all marks
- Single diacritics or bare letters collapse to plain Nodes
"""

import unicodedata

from complex_tokenization_fast._rs import FullyConnectedGraph, NodesSequence, utf8


def is_hebrew_mark(char: str) -> bool:
    return unicodedata.category(char) == "Mn"


def decompose_cluster(cluster: str):
    """Decompose a single grapheme cluster into a graph vertex."""
    base_chars = []
    marks = []

    for char in cluster:
        if is_hebrew_mark(char):
            marks.append(char)
        else:
            base_chars.append(char)

    base_text = "".join(base_chars)
    base_node = utf8(base_text) if base_text else None

    if not marks:
        if base_node is None:
            return utf8(cluster)
        return base_node

    mark_nodes = [utf8(m) for m in marks]

    if len(mark_nodes) == 1:
        diacritics = mark_nodes[0]
    else:
        diacritics = FullyConnectedGraph(nodes=tuple(mark_nodes))

    if base_node is None:
        return diacritics

    return NodesSequence(nodes=(base_node, diacritics))
