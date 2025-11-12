from functools import reduce
from typing import Counter

from complex_tokenization.draw import draw_dot_content, create_gif
from complex_tokenization.graph import GraphVertex, Node, Tree, utf8, text_to_graph, NodesSequence, GraphSettings


class Trainer:
    def __init__(self, graph=GraphVertex):
        self.graph = graph
        self.merges = []

    def train(self, num_merges: int = 100, draw=False):
        frames = []

        while True:
            if len(self.merges) >= num_merges:
                break

            if draw:
                dot_content = "\n".join(self.graph.dot())
                image = draw_dot_content(dot_content)
                frames.append(image)

            all_merges = self.graph.get_merges()
            if GraphSettings.ONLY_MINIMAL_MERGES:
                all_merges = (m for m in all_merges if all(isinstance(n, Node) for n in m))
            merges = Counter(all_merges)
            merges_compression = Counter({k: (len(k) - 1) * v for k, v in merges.items()})

            print(merges_compression.most_common(5))

            if len(merges) == 0:
                break
            nodes = merges_compression.most_common(1)[0][0]
            token = reduce(lambda x, y: x + y, nodes)

            print("Merging", token, "=", nodes)

            self.graph = self.graph.merge(token, nodes)
            self.merges.append((token, nodes))

        if draw:
            gif = create_gif(frames, save="example.gif")
            gif.show()


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

    # other_graph = text_to_graph(example_sentence)
    # example_graph = NodesSequence((example_graph, utf8(" "), other_graph))

    trainer = Trainer(graph=example_graph)
    trainer.train(num_merges=10, draw=True)
