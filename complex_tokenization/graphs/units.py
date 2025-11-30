import regex

from complex_tokenization.graph import GraphVertex, Node, NodesSequence


def characters(s: str) -> GraphVertex:
    nodes = [Node(c) for c in s]

    if len(nodes) == 1:
        return nodes[0]
    return NodesSequence(nodes=tuple(nodes))


def utf8(s: str) -> GraphVertex:
    bytes_array = s.encode("utf-8")
    nodes = [Node(bytes([b])) for b in bytes_array]
    if len(nodes) == 1:
        return nodes[0]
    return NodesSequence(nodes=tuple(nodes))


def utf8_clusters(s: str) -> GraphVertex:
    # Split string into grapheme clusters using regex
    # \X matches extended grapheme clusters
    clusters = regex.findall(r'\X', s)
    nodes = [utf8(cluster) for cluster in clusters]

    if len(nodes) == 1:
        return nodes[0]
    return NodesSequence(nodes=tuple(nodes))
