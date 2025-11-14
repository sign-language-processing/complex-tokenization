from complex_tokenization.graphs.words import pretokenize


class TestPretokenizer:
    def test_simple_english_text(self):
        """Test pretokenization of simple English text"""
        text = "hello world"
        result = pretokenize(text)
        expected = ["hello", " world"]
        assert result == expected

    def test_text_with_punctuation(self):
        """Test pretokenization with punctuation"""
        text = "hello world!"
        result = pretokenize(text)
        expected = ["hello", " world", "!"]
        assert result == expected

    def test_text_with_numbers(self):
        """Test pretokenization with numbers"""
        text = "I have 3 apples and 42 oranges"
        result = pretokenize(text)
        expected = ["I", " have", " ", "3", " apples", " and", " ", "42", " oranges"]
        assert result == expected

    def test_text_with_contractions(self):
        """Test pretokenization with contractions"""
        text = "I'm happy you're here"
        result = pretokenize(text)
        # Contractions should be preserved
        assert any("I'm" in token for token in result)
        assert any("you're" in token for token in result)

    def test_empty_string(self):
        """Test pretokenization of empty string"""
        text = ""
        result = pretokenize(text)
        assert result == []

    def test_whitespace_only(self):
        """Test pretokenization of whitespace-only text"""
        text = "   \n\n  "
        result = pretokenize(text)
        # Should tokenize whitespace
        assert len(result) > 0
        assert all(not token.strip() for token in result)

    def test_mixed_content(self):
        """Test pretokenization with mixed content: letters, numbers, punctuation"""
        text = "Hello123 world! Test 456."
        result = pretokenize(text)
        expected = ["Hello", "123", " world", "!", " Test", " ", "456", "."]
        assert result == expected

    def test_sentence_with_multiple_punctuation(self):
        """Test pretokenization of a sentence with various punctuation marks"""
        text = "Hello, world! How are you? I'm fine."
        result = pretokenize(text)
        expected = ["Hello", ",", " world", "!", " How", " are", " you", "?", " I'm", " fine", "."]
        assert result == expected

    def test_capitalized_words(self):
        """Test pretokenization with capitalized words"""
        text = "Hello World TEST"
        result = pretokenize(text)
        expected = ["Hello", " World", " TEST"]
        assert result == expected

