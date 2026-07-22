from collections import Counter, defaultdict
from functools import reduce

from complex_tokenization.draw import create_gif, draw_dot_content
from complex_tokenization.graph import GraphVertex, Tree, UnconnectedGraphs
from complex_tokenization.graphs.units import utf8


def _merge_score(item):
    # item is a (nodes, count) pair; merging a k-tuple removes k-1 nodes.
    nodes, count = item
    return (len(nodes) - 1) * count


def _index_add(total, index, i, counts):
    for merge, count in counts.items():
        total[merge] += count
        index[merge].add(i)


def _index_remove(total, index, i, counts):
    for merge, count in counts.items():
        total[merge] -= count
        if total[merge] == 0:
            del total[merge]
        others = index[merge]
        others.discard(i)
        if not others:
            del index[merge]


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

    def _steps(self, num_merges: int, progress: bool):
        remaining = range(len(self.merges), num_merges)
        if progress:
            from tqdm import tqdm
            remaining = tqdm(remaining, desc="Training", initial=len(self.merges), total=num_merges)
        return remaining

    def train(self, num_merges: int = 100, draw=False, verbose=False, progress=False, incremental=False):
        # Incremental counting only helps a forest where each merge touches few
        # subgraphs (disconnected, word-level). draw/verbose use the plain loop.
        if incremental and not draw and not verbose and isinstance(self.graph, UnconnectedGraphs):
            return self._train_incremental(num_merges, progress)

        frames = []
        for _ in self._steps(num_merges, progress):
            if draw:
                frames.append(draw_dot_content("\n".join(self.graph.dot())))

            counts = Counter(self.graph.get_merges())
            if not counts:
                break

            nodes = max(counts.items(), key=_merge_score)[0]
            if verbose:
                print("Merging", nodes, "count=", counts[nodes])
            token = reduce(lambda x, y: x + y, nodes)

            self.graph = self.graph.merge(token, nodes)
            self.merges.append((token, nodes))

        if draw:
            create_gif(frames, save="example.gif").show()

    def _train_incremental(self, num_merges: int, progress: bool = False):
        # Rebuilding Counter(graph.get_merges()) every step recounts the whole
        # forest. Instead, keep each subgraph's candidate counts plus a running
        # global total, and after a merge update only the subgraphs that
        # contained it (found via `index`). total is summed in subgraph order, so
        # picking the first max-score candidate matches max(Counter(...)) exactly.
        components = list(self.graph.subgraphs)
        comp_counts = [Counter(c.get_merges()) for c in components]
        total: dict[tuple, int] = defaultdict(int)
        index: dict[tuple, set[int]] = defaultdict(set)
        for i, counts in enumerate(comp_counts):
            _index_add(total, index, i, counts)

        for _ in self._steps(num_merges, progress):
            if not total:
                break
            best = max((len(m) - 1) * c for m, c in total.items())
            nodes = next(m for cc in comp_counts for m in cc if (len(m) - 1) * total[m] == best)
            token = reduce(lambda x, y: x + y, nodes)

            for i in list(index[nodes]):
                _index_remove(total, index, i, comp_counts[i])
                components[i] = components[i].merge(token, nodes)
                comp_counts[i] = Counter(components[i].get_merges())
                _index_add(total, index, i, comp_counts[i])

            self.merges.append((token, nodes))

        self.graph = UnconnectedGraphs(tuple(components))

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
