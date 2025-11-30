# Tokenization for Complex Scripts

This repository proposes a generic merge-based tokenization scheme, including concatenative and
non-concatenative language structures.
It therefore allows for more fitting tokenization for complex scripts (such as SignWriting and Chinese)
by decomposing words into smaller units, and representing them in various graph structures.

## Usage

Install:

```bash
git clone https://github.com/sign-language-processing/complex-tokenization.git
cd complex-tokenization
pip install ".[dev]"
```

Pretokenize text using a Huggingface Tokenizer implementation:

```python
from complex_tokenization.tokenizer import WordsSegmentationTokenizer

pretokenizer = WordsSegmentationTokenizer(max_bytes=16)
tokens = pretokenizer.tokenize("hello world! 我爱北京天安门 👩‍👩‍👧‍👦")
# ['hello ', 'world! ', '我', '爱', '北京', '天安门', ' ', '👩‍👩‍👧‍👦‍']
```

## Pretokenization

Our tokenizers run on a graph structure, which we can manipulate via pre-tokenization functions.

## Units

Units are the basic blocks we operate on, such as character, or bytes.
We implement three basic blocks:

```python
from complex_tokenization.graphs.units import characters, utf8, utf8_clusters

text = "שלום"

# Characters Split assigns a single node per character (4 characters)
assert characters(text) == NodesSequence((Node("ש"), Node("ל"), Node("ו"), Node("ם")))

# UTF-8 Split assigns a single node per byte (8 bytes)
assert utf8(text) == NodesSequence((Node(value=b'\xd7'), Node(value=b'\xa9'),
                                    Node(value=b'\xd7'), Node(value=b'\x9c'),
                                    Node(value=b'\xd7'), Node(value=b'\x95'),
                                    Node(value=b'\xd7'), Node(value=b'\x9d')))

# UTF-8 Clusters Split assigns a single node sequence per cluster, a single node per byte (4 clusters, 2 bytes each)
assert utf8_clusters(text) == NodesSequence((
    NodesSequence((Node(value=b'\xd7'), Node(value=b'\xa9'))),
    NodesSequence((Node(value=b'\xd7'), Node(value=b'\x9c'))),
    NodesSequence((Node(value=b'\xd7'), Node(value=b'\x95'))),
    NodesSequence((Node(value=b'\xd7'), Node(value=b'\x9d')))))
```

## Words

A long text that includes multiple words, can be treated as a single text (without boundaries),
or each word could be considered a single cluster.

Words can be "connected" to eachother, to allow merging over words,
or "disconnected" to disallow merging over word boundaries.

```python
from complex_tokenization.graphs.units import utf8_clusters
from complex_tokenization.graphs.words import words

text = "a few words"

# Train tokenization on the entire text
graph = utf8_clusters(text)

# Treat each word as a cluster, and words are connected

graph = words(text, units=utf8_clusters, connected=True)
```

## Tokenizers Implementation

### BNE (Byte-Ngram Encoding)

Byte-Ngram Encoding creates a merge over a sequence of units up to a certain size `N`.
It treats words as disconnected units, and does not allow merges over unmerged clusters.

```python
from complex_tokenization.graphs.settings import GraphSettings
from complex_tokenization.graphs.units import utf8_clusters
from complex_tokenization.graphs.words import words

GraphSettings.ONLY_MINIMAL_MERGES = True  # BNE only merges adjacent tokens
GraphSettings.MAX_MERGE_SIZE = N  # Maximum number of tokens to merge at a time

text = "a large text corpus..."

graph = words(text, units=utf8_clusters, connected=False)
```

### BPE (Byte-Pair Encoding)

Same as `BNE`, with a maximum of two tokens merged at a time `GraphSettings.MAX_MERGE_SIZE = 2`.

### BoundlessBPE

## Cite

If you use this code in your research, please consider citing the work:

```bibtex
@misc{moryossef2025complex,
  title={Tokenization for Complex Scripts},
  author={Moryossef, Amit},
  howpublished={\url{https://github.com/sign-language-processing/complex-tokenization}},
  year={2025}
}
```