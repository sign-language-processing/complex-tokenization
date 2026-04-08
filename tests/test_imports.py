"""Test that key classes are importable from the top-level package."""


class TestImports:
    def test_import_tokenizer(self):
        from complex_tokenization import Tokenizer
        assert Tokenizer is not None

    def test_import_bpe(self):
        from complex_tokenization import BPETokenizer
        assert BPETokenizer is not None

    def test_import_bne(self):
        from complex_tokenization import BNETokenizer
        assert BNETokenizer is not None

    def test_import_boundless(self):
        from complex_tokenization import BoundlessBPETokenizer
        assert BoundlessBPETokenizer is not None

    def test_import_super(self):
        from complex_tokenization import SuperBPETokenizer
        assert SuperBPETokenizer is not None

    def test_all_exports(self):
        import complex_tokenization
        assert len(complex_tokenization.__all__) == 5
