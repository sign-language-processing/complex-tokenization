# CLAUDE.md

## Project structure

- `complex_tokenization/` — Reference Python implementation. Simple, readable, minimal. This is the source of truth for correctness.
- `fast/` — Rust (PyO3) implementation under `complex_tokenization_fast`. Same API, same test results, optimized for speed.
- `tests/` — Shared test suite. Both implementations must pass all tests.

## Dual implementation rule

Every feature, bugfix, or API change must be implemented in **both** `complex_tokenization/` and `fast/`.
The two packages must have matching APIs and produce identical results.
However, speed and performance improvements should only be made in `fast/`, while `complex_tokenization/` should prioritize clarity and correctness.

To verify:
```bash
# Test reference
pip install -e . && pytest tests/

# Test fast (from fast/ directory)
cd fast && maturin develop --release && pytest ../tests/
```

## Testing

- Reference tests run from the repo root: `pytest tests/`
- Fast tests run from `fast/`: `pytest ../tests/` (the `fast/conftest.py` aliases `complex_tokenization` → `complex_tokenization_fast`)
- Both must pass the same 127+ tests
