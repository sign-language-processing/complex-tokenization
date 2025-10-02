# Tokenization for Complex Scripts

This repository decomposes complex scripts (such as SignWriting and Chinese) into
smaller units, and proposes more effective tokenization for NLP tasks.

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