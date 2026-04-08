# Tokenization for Complex Scripts

This repository proposes a generic merge-based tokenization scheme, including concatenative and
non-concatenative language structures.
It therefore allows for more fitting tokenization for complex scripts (such as SignWriting and Chinese)
by decomposing words into smaller units, and representing them in various graph structures.

## Usage

Install:

```bash
pip install complex-tokenization
```

Train a tokenizer:

```python
from complex_tokenization import BPETokenizer

tokenizer = BPETokenizer()
tokenizer.train(["the teacher teaches the thick thing"], num_merges=5)
print(tokenizer.get_merges())
# [(' ', 't'), ('h', 'e'), (' t', 'he'), (' t', 'e'), (' te', 'a')]
```

## Tokenizer Variants

All tokenizers accept `units`, `pretokenizer`, and variant-specific parameters:

```python
from complex_tokenization import BPETokenizer, BNETokenizer, BoundlessBPETokenizer, SuperBPETokenizer

# BPE: standard byte-pair encoding (merge_size=2, word boundaries)
tok = BPETokenizer()

# BNE: byte-ngram encoding (merge up to n tokens at once)
tok = BNETokenizer(n=4)

# Boundless BPE: merges across word boundaries
tok = BoundlessBPETokenizer()

# Super BPE: intra-word merges first, then cross-word merges
tok = SuperBPETokenizer(disconnected_merges=50)
```

## Pretokenization

By default, text is split using the GPT pretokenization regex pattern.
You can pass any HuggingFace `PreTokenizer`:

```python
from complex_tokenization import BPETokenizer
from tokenizers import Regex
from tokenizers.pre_tokenizers import Split, Whitespace

# Default: GPT regex pattern
tok = BPETokenizer()

# Whitespace splitting
tok = BPETokenizer(pretokenizer=Whitespace())

# Custom regex
tok = BPETokenizer(pretokenizer=Split(Regex(r"\w+|\S"), behavior="isolated"))
```

## Units

Units are the basic blocks we operate on.
We implement three base units, plus language-specific decompositions via the script registry:

```python
from complex_tokenization import BPETokenizer

# UTF-8 grapheme clusters (default) — one node sequence per cluster
tok = BPETokenizer(units="utf8_clusters")

# Raw UTF-8 bytes — one node per byte
tok = BPETokenizer(units="utf8")

# Characters — one node per Unicode character
tok = BPETokenizer(units="characters")
```

### Language-Specific Units

Register script-specific handlers for structural decomposition:

```python
from complex_tokenization import BPETokenizer
from complex_tokenization.languages.hebrew.decompose import decompose_cluster
from complex_tokenization.languages.chinese.graph import chinese_character_to_graph

tok = BPETokenizer()
tok.register_script("Hebrew", decompose_cluster)  # nikkud/dagesh as FullyConnectedGraph
tok.register_script("Han", chinese_character_to_graph)  # IDS tree decomposition
tok.train(texts, num_merges=100)
```

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
