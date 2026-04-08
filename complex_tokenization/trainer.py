from collections import Counter
from functools import reduce

from complex_tokenization.draw import create_gif, draw_dot_content
from complex_tokenization.graph import GraphVertex, Tree, UnconnectedGraphs
from complex_tokenization.graphs.units import utf8


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
        frames = []

        remaining = range(len(self.merges), num_merges)
        if progress:
            from tqdm import tqdm
            remaining = tqdm(remaining, desc="Training", initial=len(self.merges), total=num_merges)

        for _ in remaining:

            if draw:
                dot_content = "\n".join(self.graph.dot())
                image = draw_dot_content(dot_content)
                frames.append(image)

            counts = Counter(self.graph.get_merges())

            if not counts:
                break

            nodes = max(counts, key=lambda k: (len(k) - 1) * counts[k])

            if verbose:
                print("Merging", nodes, "count=", counts[nodes])
            token = reduce(lambda x, y: x + y, nodes)

            self.graph = self.graph.merge(token, nodes)
            self.merges.append((token, nodes))

        if draw:
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
    # example_sentence = "the teacher teaches the thick."
    example_sentence = "test test"
    # example_graph = sentence_to_graph(example_sentence)

    # other_graph = words(example_sentence)
    # example_graph = NodesSequence((example_graph, utf8(" "), other_graph))

    trainer = Trainer(graph=example_graph)
    trainer.train(num_merges=10, draw=True)
