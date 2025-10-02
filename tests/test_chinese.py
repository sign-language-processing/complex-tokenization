import json
from pathlib import Path

import pytest

from complex_tokenization.chinese.ideographic_description_sequences import (
    ids_tree_to_string,
    parse_ideographic_description_sequences,
)


class TestDictionary:
    def test_all_values_unique(self):
        """Test that all values in dictionary.json are unique"""
        dictionary_path = Path(__file__).parent.parent / "complex_tokenization" / "chinese" / "dictionary.json"

        if not dictionary_path.exists():
            pytest.skip("dictionary.json not found")

        with open(dictionary_path) as f:
            dictionary = json.load(f)

        values = list(dictionary.values())
        unique_values = set(values)

        if len(values) != len(unique_values):
            # Find duplicates
            from collections import Counter
            value_counts = Counter(values)
            duplicates = {v: count for v, count in value_counts.items() if count > 1}
            # Sort by count descending, then by value for stable ordering
            top_duplicates = sorted(duplicates.items(), key=lambda x: (-x[1], x[0]))[:20]
            top_duplicates_str = [(f"{v}: {count} occurrences "
                                   f"({'/'.join([k for k, val in dictionary.items() if val == v])})")
                                  for v, count in top_duplicates]
            pytest.fail(f"Found {len(duplicates)} ({len(values) - len(unique_values)}) duplicate values:\n"
                        + "\n".join(top_duplicates_str))


class TestIdeographicDescriptionSequences:
    def test_parse_complex_nested(self):
        """Test parsing: ⿱⿳𠂊田一⿰⿳𠂊田一⿳𠂊田一"""
        ids = "⿱⿳𠂊田一⿰⿳𠂊田一⿳𠂊田一"
        tree = parse_ideographic_description_sequences(ids)

        # Check root
        assert tree.value == "⿱"
        assert len(tree.children) == 2

        # Check first child (⿳𠂊田一)
        first_child = tree.children[0]
        assert first_child.value == "⿳"
        assert len(first_child.children) == 3
        assert first_child.children[0].value == "𠂊"
        assert first_child.children[1].value == "田"
        assert first_child.children[2].value == "一"

        # Check second child (⿰⿳𠂊田一⿳𠂊田一)
        second_child = tree.children[1]
        assert second_child.value == "⿰"
        assert len(second_child.children) == 2

        expected_tree = """└── Template: ⿱
    ├── Template: ⿳
    │   ├── Radical: 𠂊
    │   ├── Radical: 田
    │   └── Radical: 一
    └── Template: ⿰
        ├── Template: ⿳
        │   ├── Radical: 𠂊
        │   ├── Radical: 田
        │   └── Radical: 一
        └── Template: ⿳
            ├── Radical: 𠂊
            ├── Radical: 田
            └── Radical: 一
"""
        assert ids_tree_to_string(tree) == expected_tree

    def test_parse_with_surround(self):
        """Test parsing: ⿰⿳爫龴⿵冂⿱厶又乚"""
        ids = "⿰⿳爫龴⿵冂⿱厶又乚"
        tree = parse_ideographic_description_sequences(ids)

        # Check root
        assert tree.value == "⿰"
        assert len(tree.children) == 2

        expected_tree = """└── Template: ⿰
    ├── Template: ⿳
    │   ├── Radical: 爫
    │   ├── Radical: 龴
    │   └── Template: ⿵
    │       ├── Radical: 冂
    │       └── Template: ⿱
    │           ├── Radical: 厶
    │           └── Radical: 又
    └── Radical: 乚
"""
        assert ids_tree_to_string(tree) == expected_tree

    def test_parse_with_overlay(self):
        """Test parsing: ⿱⿻⿻コ一丨一"""
        ids = "⿱⿻⿻コ一丨一"
        tree = parse_ideographic_description_sequences(ids)

        # Check root
        assert tree.value == "⿱"
        assert len(tree.children) == 2

        expected_tree = """└── Template: ⿱
    ├── Template: ⿻
    │   ├── Template: ⿻
    │   │   ├── Radical: コ
    │   │   └── Radical: 一
    │   └── Radical: 丨
    └── Radical: 一
"""
        assert ids_tree_to_string(tree) == expected_tree

    def test_simple_binary(self):
        """Test simple binary IDS: ⿰木木"""
        ids = "⿰木木"
        tree = parse_ideographic_description_sequences(ids)

        assert tree.value == "⿰"
        assert len(tree.children) == 2
        assert tree.children[0].value == "木"
        assert tree.children[1].value == "木"

        expected_tree = """└── Template: ⿰
    ├── Radical: 木
    └── Radical: 木
"""
        assert ids_tree_to_string(tree) == expected_tree

    def test_empty_string_raises_error(self):
        """Test that empty string raises ValueError"""
        with pytest.raises(ValueError, match="Empty IDS string"):
            parse_ideographic_description_sequences("")

    def test_to_dict(self):
        """Test tree to dictionary conversion"""
        ids = "⿰木木"
        tree = parse_ideographic_description_sequences(ids)
        result = tree.to_dict()

        assert result["type"] == "template"
        assert result["value"] == "⿰"
        assert len(result["children"]) == 2
        assert result["children"][0]["type"] == "radical"
        assert result["children"][0]["value"] == "木"

    def test_all_dictionary_items_parseable(self):
        """Test that all items in dictionary.json are parseable"""
        dictionary_path = Path(__file__).parent.parent / "complex_tokenization" / "chinese" / "dictionary.json"

        if not dictionary_path.exists():
            pytest.skip("dictionary.json not found")

        with open(dictionary_path) as f:
            dictionary = json.load(f)

        templates = list(dictionary.values())

        for template in templates:
            parse_ideographic_description_sequences(template)
