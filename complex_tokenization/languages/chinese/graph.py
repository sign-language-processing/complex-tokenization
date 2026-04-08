"""Convert Chinese characters into graph structures using IDS decomposition."""

from complex_tokenization.graph import GraphVertex, Tree
from complex_tokenization.graphs.units import utf8
from complex_tokenization.languages.chinese.ideographic_description_sequences import (
    IDSNode,
    get_ids_for_character,
    parse_ideographic_description_sequences,
)


def ids_node_to_graph(node: IDSNode) -> GraphVertex:
    if node.is_leaf():
        return utf8(node.value)

    root = utf8(node.value)
    children = tuple(ids_node_to_graph(child) for child in node.children)
    return Tree(root=root, children=children)


def chinese_character_to_graph(cluster: str) -> GraphVertex:
    """Convert a Chinese character cluster to a graph, decomposing via IDS if possible."""
    if len(cluster) == 1:
        ids = get_ids_for_character(cluster)
        if ids is not None:
            try:
                tree = parse_ideographic_description_sequences(ids)
                return ids_node_to_graph(tree)
            except ValueError:
                pass
    return utf8(cluster)
