from collections.abc import Callable

import regex

from complex_tokenization.graph import GraphVertex, Node, NodesSequence

_cluster_handlers: dict[str, Callable[[str], GraphVertex]] = {}


def register_script(script: str, handler: Callable[[str], GraphVertex]):
    """Register a handler for grapheme clusters matching a Unicode script.

    The script name must be a valid Unicode script property (e.g. "Han",
    "Hebrew"). When utf8_clusters processes a cluster whose first character
    matches the script, the handler is called instead of the default utf8.
    """
    _cluster_handlers[script] = handler


def _get_handler(cluster: str) -> Callable[[str], GraphVertex] | None:
    if not _cluster_handlers:
        return None
    first_char = cluster[0]
    for script, handler in _cluster_handlers.items():
        if regex.match(rf'\p{{{script}}}', first_char):
            return handler
    return None


def characters(s: str) -> GraphVertex:
    nodes = [Node(c.encode("utf-8")) for c in s]

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
    clusters = regex.findall(r'\X', s)
    nodes = []
    for cluster in clusters:
        handler = _get_handler(cluster)
        if handler is not None:
            nodes.append(handler(cluster))
        else:
            nodes.append(utf8(cluster))

    if len(nodes) == 1:
        return nodes[0]
    return NodesSequence(nodes=tuple(nodes))
