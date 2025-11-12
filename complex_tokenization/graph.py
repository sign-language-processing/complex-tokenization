from dataclasses import dataclass
from typing import Iterable, Iterator

from complex_tokenization.chinese.ideographic_description_sequences import get_character_for_ids
from complex_tokenization.pretokenizer import pretokenize


def dot_escape(s: str) -> str:
    return s \
        .replace("\\", "\\\\") \
        .replace('"', '\\"') \
        .replace("\n", "\\n")


class GraphSettings:
    USE_SINGLETONS = False  # speeds up computation but hurts visualization
    MAX_MERGE_SIZE = 3
    ONLY_MINIMAL_MERGES = True


class GraphVertex:
    _instances = {}  # Singleton pattern

    def __new__(cls, *args, **kwargs):
        if not GraphSettings.USE_SINGLETONS:
            return super().__new__(cls)

        key = (args, tuple(sorted(kwargs.items())))
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
            cls._instances[key].__init__(*args, **kwargs)  # optional, if side effects desired
        return cls._instances[key]

    def __bytes__(self):
        raise NotImplementedError

    def __str__(self):
        self_str = bytes(self).decode("utf-8", errors="replace")
        token_replacement = get_character_for_ids(self_str)
        if token_replacement is not None:
            return token_replacement
        return self_str

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)

    def dot(self, level=0) -> Iterable[str]:
        raise NotImplementedError

    @property
    def oid(self) -> str:  # object pointer id for Graphviz node id
        return f"o{id(self):x}"

    def get_merges(self) -> list[str] | Iterator[tuple[str, ...]]:
        return []

    def merge(self, token, merge):
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
        num_nodes = len(self.nodes)
        for i, node in enumerate(self.nodes):
            yield from node.get_merges()

            if GraphSettings.ONLY_MINIMAL_MERGES and not isinstance(node, Node):
                break

            for j in range(i + 2, min(i + GraphSettings.MAX_MERGE_SIZE + 1, num_nodes + 1)):
                if GraphSettings.ONLY_MINIMAL_MERGES and j < num_nodes and not isinstance(self.nodes[j], Node):
                    break
                yield tuple(self.nodes[i:j])

    def merge(self, token: Node, merge: tuple[Node, ...]):
        m = len(merge)
        i = 0
        out: list[Node] = []
        nodes = self.nodes  # local alias

        while i <= len(nodes) - m:
            if tuple(nodes[i:i + m]) == merge:
                out.append(Node(value=token.value))
                i += m  # skip the matched span
            else:
                out.append(nodes[i])
                i += 1

        # append any remaining tail
        out.extend(nodes[i:])

        if len(out) == 1:
            return out[0]

        merged_nodes = tuple([n.merge(token, merge) for n in out])
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
            print("other", type(other))
            print(self.nodes, type(self.nodes))
            return NodesSequence(self.nodes + tuple([other]))


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


def utf8(s: str) -> NodesSequence:
    bytes_array = s.encode("utf-8")
    return Node(bytes_array)
    nodes = [Node(bytes([b])) for b in bytes_array] # TODO each grapheme cluster should be a list
    if len(nodes) == 1:
        return nodes[0]
    return NodesSequence(nodes=tuple(nodes))


def text_to_graph(text: str) -> NodesSequence:
    words = pretokenize(text)
    nodes = [utf8(word) for word in words]
    if len(nodes) == 1:
        return nodes[0]
    return NodesSequence(nodes=tuple(nodes))
