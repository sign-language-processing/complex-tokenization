import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from functools import wraps
from itertools import chain

from complex_tokenization.graphs.settings import GraphSettings
from complex_tokenization.languages.chinese.ideographic_description_sequences import get_character_for_ids


def bytes_to_str(data: bytes) -> str:
    """Lossless textual form of token bytes: valid UTF-8 reads as itself,
    undecodable bytes become \\xNN escapes. Literal backslashes are doubled
    first, so str_to_bytes can always invert exactly."""
    return data.replace(b"\\", b"\\\\").decode("utf-8", errors="backslashreplace")


def str_to_bytes(s: str) -> bytes:
    return re.sub(
        rb"\\\\|\\x([0-9a-fA-F]{2})",
        lambda m: bytes([int(m[1], 16)]) if m[1] else b"\\",
        s.encode("utf-8"),
    )


def dot_escape(s: str) -> str:
    return s \
        .replace("\\", "\\\\") \
        .replace('"', '\\"') \
        .replace("\n", "\\n")


def merge_shared(merge_fn):
    """Merge each distinct subgraph object once per top-level merge call.

    Duplicated subgraphs are shared objects (the build cache). The memo, shared
    down the recursion, maps id(subgraph) -> merged result, so duplicates are
    merged once and keep sharing one object (and one get_merges memo) at any
    nesting depth, instead of diverging into equal-but-distinct copies.
    """
    @wraps(merge_fn)
    def merge(self, token, nodes, memo=None):
        if memo is None:
            memo = {}
        result = memo.get(id(self))
        if result is None:
            result = memo[id(self)] = merge_fn(self, token, nodes, memo)
        return result
    return merge




class GraphVertex:
    # Empty slots so the slotted subclasses below actually suppress __dict__;
    # without this, a non-slotted base hands every node an unused __dict__.
    __slots__ = ()

    def __bytes__(self):
        raise NotImplementedError

    def __str__(self):
        self_str = bytes_to_str(bytes(self))
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

    def merge(self, token, merge, memo=None) -> "GraphVertex":
        raise NotImplementedError

    def node_count(self) -> int:
        raise NotImplementedError


class Node(bytes, GraphVertex):
    # A Node *is* its bytes, so __hash__/__eq__/__len__ are bytes' C-level
    # operations — which is what the trainer's Counter and merge scans hammer.
    # bytes wins the MRO for those, but we still want GraphVertex's __str__.
    __str__ = GraphVertex.__str__

    def __new__(cls, value: bytes):
        return super().__new__(cls, value)

    def __bytes__(self):
        return self[:]  # a plain bytes copy (not the Node subclass)

    def dot(self, level=0) -> Iterable[str]:
        yield "\t" * level + f'{self.oid} [label="{dot_escape(str(self))}"];'

    def merge(self, token: "Node", merge: tuple, memo=None):
        return self

    def node_count(self) -> int:
        return 1

    def __add__(self, other):
        if isinstance(other, NodesSequence):
            return NodesSequence((self,) + other.nodes)
        return Node(b"".join((self, other)))  # both are bytes; join avoids Node.__add__ recursion


@dataclass(frozen=True, slots=True)
class NodesSequence(GraphVertex):
    nodes: tuple[GraphVertex, ...]
    # memo slot for get_merges; excluded from eq/hash/repr (see get_merges)
    _merges: tuple | None = field(default=None, init=False, compare=False, repr=False)

    def __bytes__(self):
        buffer = bytearray()
        for node in self.nodes:
            buffer += bytes(node)
        return bytes(buffer)

    @property
    def oid(self) -> str:  # object pointer id for Graphviz node id
        return self.nodes[0].oid

    def get_merges(self):
        if not GraphSettings.TRADE_MEMORY_FOR_SPEED:
            return self._iter_merges()
        # Memoize: get_merges is a pure function of an immutable node, and merge
        # returns self for unchanged subtrees, so the same objects recur across
        # merges and a full re-walk becomes a cache hit. Valid while GraphSettings
        # is fixed, which holds for a node's lifetime during training.
        cached = self._merges
        if cached is None:
            cached = tuple(self._iter_merges())
            object.__setattr__(self, "_merges", cached)
        return cached

    def _iter_merges(self):
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

    def node_count(self) -> int:
        return sum(n.node_count() for n in self.nodes)

    @merge_shared
    def merge(self, token: Node, merge: tuple["GraphVertex", ...], memo: dict):
        nodes = self.nodes

        # _merges (when memoized) lists every mergeable subsequence in this
        # subtree, so if the merge isn't among them nothing here changes.
        cached = self._merges
        if cached is not None and merge not in cached:
            return self

        m = len(merge)
        n = len(nodes)
        first = merge[0]
        out: list[GraphVertex] = []
        i = 0
        while i <= n - m:
            # First-node guard before slicing (see #29): skips the nodes[i:i + m]
            # allocation at the many positions that can't start the merge.
            if nodes[i] == first and nodes[i:i + m] == merge:
                out.append(token)
                i += m
            else:
                out.append(nodes[i])
                i += 1
        out.extend(nodes[i:])

        if len(out) == 1:
            return out[0]
        merged_nodes = tuple(node.merge(token, merge, memo) for node in out)
        if merged_nodes == nodes:
            return self
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

    def node_count(self) -> int:
        return self.root.node_count() + sum(c.node_count() for c in self.children)

    @merge_shared
    def merge(self, token: Node, nodes: tuple, memo: dict):
        if nodes[0] == self.root:
            if len(nodes) == len(self.children) + 1:
                if all(nodes[i + 1] == child for i, child in enumerate(self.children)):
                    return token

        root = self.root.merge(token, nodes, memo)
        children = tuple(child.merge(token, nodes, memo) for child in self.children)
        if root == self.root and children == self.children:
            return self
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

    def node_count(self) -> int:
        return sum(n.node_count() for n in self.nodes)

    @merge_shared
    def merge(self, token: Node, merge: tuple, memo: dict):
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

        merged_nodes = tuple(n.merge(token, merge, memo) for n in self.nodes)
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

    def __post_init__(self):
        # Flatten nested UnconnectedGraphs so all subgraphs are at one level.
        # Skip the rebuild when nothing is nested (the common case, e.g. every
        # merge step). Uses object.__setattr__ because the dataclass is frozen.
        if not any(isinstance(sg, UnconnectedGraphs) for sg in self.subgraphs):
            return
        flat = []
        for sg in self.subgraphs:
            if isinstance(sg, UnconnectedGraphs):
                flat.extend(sg.subgraphs)
            else:
                flat.append(sg)
        object.__setattr__(self, 'subgraphs', tuple(flat))

    def __bytes__(self):
        raise Exception("Cannot convert UnconnectedGraphs to bytes")

    def node_count(self) -> int:
        return sum(sg.node_count() for sg in self.subgraphs)

    @merge_shared
    def merge(self, token: Node, merge: tuple, memo: dict):
        old = self.subgraphs
        new = tuple(sg.merge(token, merge, memo) for sg in old)
        if new == old:
            return self
        return UnconnectedGraphs(subgraphs=new)

    def get_merges(self) -> Iterator[tuple]:
        return chain.from_iterable(sg.get_merges() for sg in self.subgraphs)

    def dot(self, level=0) -> Iterable[str]:
        for subgraph in self.subgraphs:
            yield from subgraph.dot(level)

