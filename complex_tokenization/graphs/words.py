from collections.abc import Iterable

import regex as re

from complex_tokenization.graph import GraphVertex, NodesSequence, UnconnectedGraphs
from complex_tokenization.graphs.units import utf8_clusters

# From openai/gpt-oss-20b
pattern = (
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    "|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    "|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
)


def pretokenize(text: str) -> Iterable[str]:
    return [match.group(0) for match in re.finditer(pattern, text)]


def words(text: str, connected=True, units=utf8_clusters) -> GraphVertex:
    tokens = pretokenize(text)
    nodes = [units(word) for word in tokens]
    if len(nodes) == 1:
        return nodes[0]
    if connected:
        return NodesSequence(nodes=tuple(nodes))
    return UnconnectedGraphs(subgraphs=tuple(nodes))
