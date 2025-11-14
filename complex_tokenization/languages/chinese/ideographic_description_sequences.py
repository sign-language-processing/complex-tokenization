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

from dataclasses import dataclass
from functools import cache
import json
from pathlib import Path

# IDCs that take 2 components
BINARY_IDCS = set('⿰⿱⿴⿵⿶⿷⿸⿹⿺⿻')

# IDCs that take 3 components
TERNARY_IDCS = set('⿲⿳')

# All IDCs
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
        """Check if this node is a leaf (radical)."""
        return len(self.children) == 0

    def is_template(self) -> bool:
        """Check if this node is a template (IDC)."""
        return self.value in ALL_IDCS

    def to_dict(self):
        """Convert to dictionary representation."""
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
        ids: The IDS string to parse (e.g., "⿱⿳𠂊田一⿰⿳𠂊田一⿳𠂊田一")

    Returns:
        IDSNode: Root node of the parsed tree

    Example:
        >>> tree = parse_ideographic_description_sequences("⿱⿳𠂊田一⿰⿳𠂊田一⿳𠂊田一")
        >>> tree.value
        '⿱'
        >>> len(tree.children)
        2
    """
    if not ids:
        raise ValueError("Empty IDS string")

    index = [0]  # Use list to maintain reference in nested function

    def parse_node() -> IDSNode:
        if index[0] >= len(ids):
            raise ValueError(f"Unexpected end of IDS string at position {index[0]}")

        char = ids[index[0]]
        index[0] += 1

        if char in TERNARY_IDCS:
            # Parse 3 children
            node = IDSNode(value=char)
            node.children = [parse_node() for _ in range(3)]
            return node
        elif char in BINARY_IDCS:
            # Parse 2 children
            node = IDSNode(value=char)
            node.children = [parse_node() for _ in range(2)]
            return node
        else:
            # Leaf node (radical)
            return IDSNode(value=char)

    root = parse_node()

    # Verify we consumed the entire string
    if index[0] < len(ids):
        raise ValueError(f"Extra characters after parsing: {ids[index[0]:]}")

    return root


def ids_tree_to_string(node: IDSNode, prefix: str = "", is_last: bool = True) -> str:
    """
    Convert an IDS tree to a readable ASCII tree representation.

    Args:
        node: The root node of the tree
        prefix: Current line prefix for drawing branches
        is_last: Whether this is the last child of its parent

    Returns:
        String representation of the tree
    """
    # Current node connector
    connector = "└── " if is_last else "├── "

    # Node label
    if node.is_leaf():
        label = f"Radical: {node.value}"
    else:
        label = f"Template: {node.value}"

    result = prefix + connector + label + "\n"

    # Process children
    if not node.is_leaf():
        # Extension for children's prefix
        extension = "    " if is_last else "│   "
        new_prefix = prefix + extension

        for i, child in enumerate(node.children):
            is_last_child = (i == len(node.children) - 1)
            result += ids_tree_to_string(child, new_prefix, is_last_child)

    return result

@cache
def load_characters_dictionary():
    """Load the IDS dictionary from the JSON file."""
    dictionary_path = Path(__file__).parent / "dictionary.json"
    with open(dictionary_path, encoding='utf-8') as f:
        dictionary = json.load(f)
    return dictionary

@cache
def reversed_characters_dictionary():
    """Load the reversed IDS dictionary from the JSON file."""
    dictionary = load_characters_dictionary()
    reversed_dict = {v: k for k, v in dictionary.items()}
    return reversed_dict


def get_ids_for_character(char: str) -> str:
    """Get the IDS for a given character from the dictionary."""
    dictionary = load_characters_dictionary()
    return dictionary.get(char)

def get_character_for_ids(ids: str) -> str | None:
    """Get the character for a given IDS from the reversed dictionary."""
    reversed_dict = reversed_characters_dictionary()
    return reversed_dict.get(ids, None)