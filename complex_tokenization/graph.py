from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from complex_tokenization.graphs.settings import GraphSettings
from complex_tokenization.languages.chinese.ideographic_description_sequences import get_character_for_ids


def dot_escape(s: str) -> str:
    return s \
        .replace("\\", "\\\\") \
        .replace('"', '\\"') \
        .replace("\n", "\\n")




class GraphVertex:
    def __bytes__(self):
        raise NotImplementedError

    def __str__(self):
        self_str = bytes(self).decode("utf-8", errors="replace")
        token_replacement = get_character_for_ids(self_str)
        if token_replacement is not None:
            return token_replacement
        return self_str

    def dot(self, level=0) -> Iterable[str]:
        raise NotImplementedError

    @property
    def oid(self) -> str:  # object pointer id for Graphviz node id
        return f"o{id(self):x}"

    def get_merges(self) -> list[str] | Iterator[tuple[str, ...]]:
        return []

    def merge(self, token, merge) -> "GraphVertex":
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Node(GraphVertex):
    value: bytes

    def __bytes__(self):
        return self.value

    def dot(self, level=0) -> Iterable[str]:
        yield "\t" * level + f'{self.oid} [label="{dot_escape(str(self))}"];'

    def merge(self, token: "Node", merge: tuple):
        return self

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.value == other.value

    def __add__(self, other):
        if isinstance(other, NodesSequence):
            return NodesSequence(tuple([self]) + other.nodes)
        return Node(value=self.value + other.value)

    def __len__(self):
        return len(self.value)


@dataclass(frozen=True, slots=True)
class NodesSequence(GraphVertex):
    nodes: tuple[GraphVertex, ...]

    def __bytes__(self):
        buffer = bytearray()
        for node in self.nodes:
            buffer += bytes(node)
        return bytes(buffer)

    @property
    def oid(self) -> str:  # object pointer id for Graphviz node id
        return self.nodes[0].oid

    def get_merges(self):
        nodes = self.nodes
        num_nodes = len(nodes)
        only_minimal = GraphSettings.ONLY_MINIMAL_MERGES
        max_size = GraphSettings.MAX_MERGE_SIZE

        for i in range(num_nodes):
            node = nodes[i]
            yield from node.get_merges()

            if only_minimal and not isinstance(node, Node):
                continue

            for j in range(i + 2, min(i + max_size + 1, num_nodes + 1)):
                if only_minimal and not isinstance(nodes[j - 1], Node):
                    break
                yield (nodes[i], nodes[j - 1]) if j - i == 2 else tuple(nodes[i:j])

    def merge(self, token: Node, merge: tuple["GraphVertex", ...]):
        m = len(merge)
        nodes = self.nodes
        n = len(nodes)
        out: list[GraphVertex] = []
        i = 0

        while i <= n - m:
            if nodes[i:i + m] == merge:
                out.append(token)
                i += m
            else:
                out.append(nodes[i])
                i += 1
        out.extend(nodes[i:])

        if len(out) == 1:
            return out[0]

        merged_nodes = tuple(n.merge(token, merge) for n in out)
        return NodesSequence(merged_nodes)

    def dot(self, level=0) -> Iterable[str]:
        color = "grey" if level % 2 == 1 else "lightgrey"

        # create a subgraph to group nodes
        yield f"subgraph cluster_{id(self)} {{"
        yield f'\tlabel="{str(self)}";'
        yield f'\tstyle=filled; color="{color}";'
        yield '\tnode [style=filled, color=white];'
        yield ''
        yield '\tedge [arrowhead=none];'
        yield ''
        last_node = None
        for node in self.nodes:
            yield f'\t{"".join(node.dot(level + 1))}'
            if last_node is not None:
                yield f'\t{last_node.oid} -> {node.oid};'
            last_node = node
        yield ''
        yield "}"

    def __add__(self, other):
        if isinstance(other, NodesSequence):
            return NodesSequence(self.nodes + other.nodes)
        if isinstance(other, Node):
            return NodesSequence(self.nodes + (other,))


@dataclass(frozen=True, slots=True)
class Tree(GraphVertex):
    root: GraphVertex
    children: tuple[GraphVertex, ...]

    def dot(self, level=0) -> Iterable[str]:
        color = "#cce5ff" if level % 2 == 1 else "lightblue"

        yield "subgraph cluster_" + self.oid + " {"
        yield f'\tlabel="{dot_escape(str(self))}";'
        yield f'\tstyle=filled; color="{color}";'
        yield '\tnode [style=filled, color=white];'
        yield '\tedge [arrowhead=normal];'
        yield ''
        yield from self.root.dot(level + 1)
        for i, child in enumerate(self.children, start=1):
            yield from ("\t" + line for line in child.dot(level + 1))
            yield f'\t{self.oid} -> {child.oid} [label="{i}"];'
        yield ''
        yield '\tedge [arrowhead=none];'
        yield "}"

    @property
    def oid(self, level=0) -> str:
        return self.root.oid

    def get_merges(self) -> Iterator[tuple]:
        yield from self.root.get_merges()
        if not GraphSettings.ONLY_MINIMAL_MERGES or (
            isinstance(self.root, Node) and all(isinstance(c, Node) for c in self.children)
        ):
            yield (self.root,) + self.children
        for child in self.children:
            yield from child.get_merges()

    def merge(self, token: Node, nodes: tuple):
        if nodes[0] == self.root:
            if len(nodes) == len(self.children) + 1:
                if all(nodes[i + 1] == child for i, child in enumerate(self.children)):
                    return Node(value=token.value)

        root = self.root.merge(token, nodes)
        children = tuple(child.merge(token, nodes) for child in self.children)
        return Tree(root=root, children=children)

    def __bytes__(self):
        self_bytes = bytes(self.root)
        for child in self.children:
            self_bytes += bytes(child)
        return self_bytes


@dataclass(frozen=True, slots=True)
class FullyConnectedGraph(GraphVertex):
    """A set of nodes where every pair is a valid merge candidate.

    Used for Hebrew diacritics: dagesh, nikkud, and cantillation marks
    on the same letter are interchangeable in merge order.
    """
    nodes: tuple[GraphVertex, ...]

    def __bytes__(self):
        return b"".join(bytes(n) for n in self.nodes)

    @property
    def oid(self) -> str:
        return self.nodes[0].oid

    def get_merges(self) -> Iterator[tuple]:
        for node in self.nodes:
            yield from node.get_merges()
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes)):
                if i != j:
                    yield (self.nodes[i], self.nodes[j])

    def merge(self, token: Node, merge: tuple):
        remaining = list(self.nodes)
        if len(merge) == 2:
            m0, m1 = merge
            for i in range(len(remaining)):
                if remaining[i] == m0:
                    for j in range(len(remaining)):
                        if i != j and remaining[j] == m1:
                            merged = [n for k, n in enumerate(remaining) if k not in (i, j)]
                            merged.append(token)
                            if len(merged) == 1:
                                return merged[0]
                            return FullyConnectedGraph(nodes=tuple(merged))

        merged_nodes = tuple(n.merge(token, merge) for n in self.nodes)
        if merged_nodes == self.nodes:
            return self
        return FullyConnectedGraph(nodes=merged_nodes)

    def dot(self, level=0) -> Iterable[str]:
        color = "#ffe0cc" if level % 2 == 1 else "#ffd0b0"
        yield f"subgraph cluster_{id(self)} {{"
        yield f'\tlabel="{dot_escape(str(self))}";'
        yield f'\tstyle=filled; color="{color}";'
        yield '\tnode [style=filled, color=white];'
        yield '\tedge [arrowhead=none, style=dashed];'
        yield ''
        for node in self.nodes:
            yield from node.dot(level + 1)
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                yield f'\t{self.nodes[i].oid} -> {self.nodes[j].oid} [dir=both];'
        yield "}"


@dataclass(frozen=True, slots=True)
class UnconnectedGraphs(GraphVertex):
    subgraphs: tuple[GraphVertex, ...]

    def __bytes__(self):
        raise Exception("Cannot convert UnconnectedGraphs to bytes")

    def merge(self, token: Node, merge: tuple):
        old = self.subgraphs
        new = tuple(sg.merge(token, merge) for sg in old)
        if new == old:
            return self
        return UnconnectedGraphs(subgraphs=new)

    def get_merges(self) -> Iterator[tuple]:
        for subgraph in self.subgraphs:
            yield from subgraph.get_merges()

    def dot(self, level=0) -> Iterable[str]:
        for subgraph in self.subgraphs:
            yield from subgraph.dot(level)

