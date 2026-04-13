from tokenizers import Regex
from tokenizers.pre_tokenizers import Split

from complex_tokenization_fast._rs import NodesSequence, UnconnectedGraphs
from complex_tokenization_fast.graphs.units import utf8_clusters

GPT_PATTERN = (
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    "|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    "|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
)
GPTPretokenizer = Split(Regex(GPT_PATTERN), behavior="isolated")


def pretokenize(text, pretokenizer=GPTPretokenizer):
    return [token for token, _ in pretokenizer.pre_tokenize_str(text)]


def words(text, connected=True, units=utf8_clusters, pretokenizer=GPTPretokenizer):
    tokens = pretokenize(text, pretokenizer)
    nodes = [units(word) for word in tokens]
    if len(nodes) == 1:
        return nodes[0]
    if connected:
        return NodesSequence(tuple(nodes))
    return UnconnectedGraphs(tuple(nodes))
