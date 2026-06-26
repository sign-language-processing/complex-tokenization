from collections import Counter, defaultdict
from functools import reduce

from complex_tokenization.draw import create_gif, draw_dot_content
from complex_tokenization.graph import GraphVertex, Tree, UnconnectedGraphs
from complex_tokenization.graphs.units import utf8


class _TrainState:
    """Incremental candidate bookkeeping for a forest of independent components.

    A merge only affects the components that contain it, so instead of recounting
    the whole graph each step we cache per-component candidate counts and keep a
    running global total in sync. Identical components are deduplicated (every
    merge applies to the whole graph, so they stay identical) and tracked with a
    multiplicity; this leaves counts and first-appearance order unchanged.
    """

    def __init__(self, raw: list[GraphVertex]):
        self.components: list[GraphVertex] = []
        self.weights: list[int] = []
        self.raw_to_unique: list[int] = []
        seen: dict[GraphVertex, int] = {}
        for c in raw:
            i = seen.get(c)
            if i is None:
                i = len(self.components)
                seen[c] = i
                self.components.append(c)
                self.weights.append(1)
            else:
                self.weights[i] += 1
            self.raw_to_unique.append(i)

        self.comp_counts = [Counter(c.get_merges()) for c in self.components]
        # total: running global candidate counts.
        # index: candidate -> components containing it, to reach changed
        # components without scanning the whole forest.
        self.total: dict[tuple, int] = defaultdict(int)
        self.index: dict[tuple, set[int]] = defaultdict(set)
        for i, counts in enumerate(self.comp_counts):
            self._add(i, counts)

    def _add(self, i: int, counts: Counter):
        weight = self.weights[i]
        for merge, count in counts.items():
            self.total[merge] += count * weight
            self.index[merge].add(i)

    def _remove(self, i: int, counts: Counter):
        weight = self.weights[i]
        for merge, count in counts.items():
            self.total[merge] -= count * weight
            if self.total[merge] == 0:
                del self.total[merge]
            others = self.index[merge]
            others.discard(i)
            if not others:
                del self.index[merge]

    def apply_merge(self, i: int, token, nodes: tuple):
        self._remove(i, self.comp_counts[i])
        self.components[i] = self.components[i].merge(token, nodes)
        self.comp_counts[i] = Counter(self.components[i].get_merges())
        self._add(i, self.comp_counts[i])

    def merged_graph(self) -> list[GraphVertex]:
        return [self.components[i] for i in self.raw_to_unique]


def _pick_best(total: dict[tuple, int], comp_counts: list) -> tuple | None:
    # Highest score (len - 1) * count, ties broken by first appearance in
    # traversal order. A Counter iterates keys in get_merges order, so scanning
    # components in order and returning the first max-score candidate reproduces
    # max(Counter(graph.get_merges()), key=score) exactly.
    if not total:
        return None
    max_score = max((len(k) - 1) * c for k, c in total.items())
    for counts in comp_counts:
        for nodes in counts:
            if (len(nodes) - 1) * total[nodes] == max_score:
                return nodes
    return None


class Trainer:
    def __init__(self, graph: GraphVertex | None = None, graphs: tuple[GraphVertex, ...] | None = None):
        if graphs is None and graph is None:
            raise ValueError("Must provide either graph or graphs")
        if graphs is not None and graph is not None:
            raise ValueError("Must provide either graph or graphs, not both")

        if graphs is not None:
            graph = UnconnectedGraphs(graphs)

        self.graph = graph
        self.merges = []

    def _components(self) -> list[GraphVertex]:
        # The graph is a forest of independent components: a merge only ever
        # affects the components that contain it. Splitting lets us recount and
        # re-merge just those, instead of walking the whole graph every step.
        if isinstance(self.graph, UnconnectedGraphs):
            return list(self.graph.subgraphs)
        return [self.graph]

    def _store_components(self, components: list[GraphVertex]):
        if isinstance(self.graph, UnconnectedGraphs):
            self.graph = UnconnectedGraphs(tuple(components))
        else:
            self.graph = components[0]

    def train(self, num_merges: int = 100, draw=False, verbose=False, progress=False):
        if draw:
            return self._train_draw(num_merges, verbose)
        if len(self.merges) >= num_merges:
            return

        state = _TrainState(self._components())

        remaining = range(len(self.merges), num_merges)
        if progress:
            from tqdm import tqdm
            remaining = tqdm(remaining, desc="Training", initial=len(self.merges), total=num_merges)

        for _ in remaining:
            nodes = _pick_best(state.total, state.comp_counts)
            if nodes is None:
                break

            if verbose:
                print("Merging", nodes, "count=", state.total[nodes])
            token = reduce(lambda x, y: x + y, nodes)

            for i in list(state.index[nodes]):
                state.apply_merge(i, token, nodes)

            self.merges.append((token, nodes))

        self._store_components(state.merged_graph())

    def _train_draw(self, num_merges: int, verbose: bool):
        frames = []
        for _ in range(len(self.merges), num_merges):
            dot_content = "\n".join(self.graph.dot())
            frames.append(draw_dot_content(dot_content))

            counts = Counter(self.graph.get_merges())
            if not counts:
                break

            nodes = max(counts, key=lambda k: (len(k) - 1) * counts[k])
            if verbose:
                print("Merging", nodes, "count=", counts[nodes])
            token = reduce(lambda x, y: x + y, nodes)

            self.graph = self.graph.merge(token, nodes)
            self.merges.append((token, nodes))

        gif = create_gif(frames, save="example.gif")
        gif.show()

    def get_merges(self):
        return [tuple(str(node) for node in nodes) for _, nodes in self.merges]


if __name__ == "__main__":
    example_graph = Tree(root=utf8("⿱"), children=(
        utf8("十"),
        Tree(root=utf8("⿱"), children=(
            utf8("乛"),
            utf8("头"),
        )),
    ))
    trainer = Trainer(graph=example_graph)
    trainer.train(num_merges=10, draw=True)
