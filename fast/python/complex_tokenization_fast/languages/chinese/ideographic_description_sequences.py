"""
Parser for Ideographic Description Sequences (IDS).

IDS uses Ideographic Description Characters (IDC) to describe the structure of Chinese characters.
IDCs range from U+2FF0 (⿰) to U+2FFB (⿻).

Binary IDCs (take 2 components):
⿰ (U+2FF0) - left to right
⿱ (U+2FF1) - above to below
⿲ (U+2FF2) - left to middle to right
⿳ (U+2FF3) - above to middle to below
⿴ (U+2FF4) - surround from above
⿵ (U+2FF5) - surround from below
⿶ (U+2FF6) - surround from left
⿷ (U+2FF7) - surround from upper left
⿸ (U+2FF8) - surround from upper right
⿹ (U+2FF9) - surround from lower left
⿺ (U+2FFA) - overlaid

Ternary IDCs (take 3 components):
⿲ (U+2FF2) - left to middle to right
⿳ (U+2FF3) - above to middle to below
"""

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

BINARY_IDCS = set('⿰⿱⿴⿵⿶⿷⿸⿹⿺⿻')

TERNARY_IDCS = set('⿲⿳')

ALL_IDCS = BINARY_IDCS | TERNARY_IDCS


@dataclass
class IDSNode:
    """Represents a node in the IDS tree."""
    value: str
    children: list['IDSNode'] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_template(self) -> bool:
        return self.value in ALL_IDCS

    def to_dict(self):
        if self.is_leaf():
            return {"type": "radical", "value": self.value}
        else:
            return {
                "type": "template",
                "value": self.value,
                "children": [child.to_dict() for child in self.children]
            }


def parse_ideographic_description_sequences(ids: str) -> IDSNode:
    """
    Parse an Ideographic Description Sequence into a tree structure.

    Args:
        ids: The IDS string to parse.

    Returns:
        IDSNode: Root node of the parsed tree

    Example:
        >>> ids = "\u2ff1\u2ff3\U000200CA\u7530\u4e00\u2ff0\u2ff3\U000200CA\u7530\u4e00\u2ff3\U000200CA\u7530\u4e00"
        >>> tree = parse_ideographic_description_sequences(ids)
        >>> tree.value
        '\u2ff1'
        >>> len(tree.children)
        2
    """
    if not ids:
        raise ValueError("Empty IDS string")

    index = [0]

    def parse_node() -> IDSNode:
        if index[0] >= len(ids):
            raise ValueError(f"Unexpected end of IDS string at position {index[0]}")

        char = ids[index[0]]
        index[0] += 1

        if char in TERNARY_IDCS:
            node = IDSNode(value=char)
            node.children = [parse_node() for _ in range(3)]
            return node
        elif char in BINARY_IDCS:
            node = IDSNode(value=char)
            node.children = [parse_node() for _ in range(2)]
            return node
        else:
            return IDSNode(value=char)

    root = parse_node()

    if index[0] < len(ids):
        raise ValueError(f"Extra characters after parsing: {ids[index[0]:]}")

    return root


def ids_tree_to_string(node: IDSNode, prefix: str = "", is_last: bool = True) -> str:
    connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "

    if node.is_leaf():
        label = f"Radical: {node.value}"
    else:
        label = f"Template: {node.value}"

    result = prefix + connector + label + "\n"

    if not node.is_leaf():
        extension = "    " if is_last else "\u2502   "
        new_prefix = prefix + extension

        for i, child in enumerate(node.children):
            is_last_child = (i == len(node.children) - 1)
            result += ids_tree_to_string(child, new_prefix, is_last_child)

    return result

@cache
def load_characters_dictionary():
    dictionary_path = Path(__file__).parent / "dictionary.json"
    with open(dictionary_path, encoding='utf-8') as f:
        dictionary = json.load(f)
    return dictionary

@cache
def reversed_characters_dictionary():
    dictionary = load_characters_dictionary()
    reversed_dict = {v: k for k, v in dictionary.items()}
    from complex_tokenization_fast._rs import set_ids_reverse_dict_py
    set_ids_reverse_dict_py(reversed_dict)
    return reversed_dict


def get_ids_for_character(char: str) -> str:
    dictionary = load_characters_dictionary()
    return dictionary.get(char)

def get_character_for_ids(ids: str) -> str | None:
    reversed_dict = reversed_characters_dictionary()
    return reversed_dict.get(ids, None)
