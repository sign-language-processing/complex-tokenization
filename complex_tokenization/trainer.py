from collections import Counter
from functools import reduce

from complex_tokenization.draw import create_gif, draw_dot_content
from complex_tokenization.graph import GraphVertex, Tree, UnconnectedGraphs
from complex_tokenization.graphs.units import utf8


def _merge_score(item):
    # item is a (nodes, count) pair; merging a k-tuple removes k-1 nodes.
    nodes, count = item
    return (len(nodes) - 1) * count


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

    def train(self, num_merges: int = 100, draw=False, verbose=False, progress=False):
        remaining = range(len(self.merges), num_merges)
        if progress:
            from tqdm import tqdm
            remaining = tqdm(remaining, desc="Training", initial=len(self.merges), total=num_merges)

        frames = []
        for _ in remaining:
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
