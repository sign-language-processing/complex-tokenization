"""Make `import complex_tokenization` resolve to `complex_tokenization_fast` during tests."""
import importlib
import sys

import complex_tokenization_fast

# Eagerly import all submodules so the aliases work
submodules = [
    "graph", "graphs", "graphs.settings", "graphs.units", "graphs.words",
    "tokenizer", "trainer",
    "languages", "languages.chinese", "languages.chinese.graph",
    "languages.chinese.ideographic_description_sequences",
    "languages.hebrew", "languages.hebrew.decompose",
]
for sub in submodules:
    importlib.import_module(f"complex_tokenization_fast.{sub}")

# Alias so all test imports like `from complex_tokenization import ...` use the fast version
sys.modules["complex_tokenization"] = complex_tokenization_fast
for sub in submodules:
    sys.modules[f"complex_tokenization.{sub}"] = sys.modules[f"complex_tokenization_fast.{sub}"]
sys.modules["complex_tokenization._rs"] = complex_tokenization_fast._rs
