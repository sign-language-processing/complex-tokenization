"""
Analyze frequency of near-leaf node patterns in Chinese characters from Wikipedia data.

This script:
1. Downloads Chinese Wikipedia dataset from HuggingFace
2. Extracts and counts all Chinese characters
3. Decomposes characters using Ideographic Description Sequences (IDS)
4. Identifies near-leaf nodes (nodes whose children are all leaves)
5. Counts frequency of near-leaf patterns weighted by character frequency
"""

from collections import Counter

from datasets import load_dataset
from tqdm import tqdm

from complex_tokenization.languages.chinese.ideographic_description_sequences import (
    IDSNode,
    get_character_for_ids,
    get_ids_for_character,
    parse_ideographic_description_sequences,
)


def is_chinese_character(char: str) -> bool:
    """Check if a character is a Chinese character (CJK Unified Ideographs)."""
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF or  # CJK Unified Ideographs
        0x3400 <= code <= 0x4DBF or  # CJK Unified Ideographs Extension A
        0x20000 <= code <= 0x2A6DF or  # CJK Unified Ideographs Extension B
        0x2A700 <= code <= 0x2B73F or  # CJK Unified Ideographs Extension C
        0x2B740 <= code <= 0x2B81F or  # CJK Unified Ideographs Extension D
        0x2B820 <= code <= 0x2CEAF or  # CJK Unified Ideographs Extension E
        0x2CEB0 <= code <= 0x2EBEF or  # CJK Unified Ideographs Extension F
        0x30000 <= code <= 0x3134F     # CJK Unified Ideographs Extension G
    )


def extract_chinese_characters(text: str) -> list[str]:
    """Extract all Chinese characters from text."""
    return [char for char in text if is_chinese_character(char)]


def linearize_preorder(node: IDSNode) -> tuple[str, ...]:
    """
    Linearize a subtree in preorder (node, then children).
    Returns a tuple of values.
    """
    if node.is_leaf():
        return (node.value,)

    result = [node.value]
    for child in node.children:
        result.extend(linearize_preorder(child))
    return tuple(result)


def find_all_subtree_patterns(node: IDSNode) -> list[tuple[str, ...]]:
    """
    Find all non-leaf subtrees in the tree and linearize them in preorder.
    Returns a list of tuples representing each subtree's preorder traversal.
    """
    patterns = []

    def traverse(node: IDSNode):
        if node.is_leaf():
            return

        # Linearize this subtree
        if all(child.is_leaf() for child in node.children):
            pattern = linearize_preorder(node)
            patterns.append(pattern)

        # Continue traversing to find all subtrees
        for child in node.children:
            traverse(child)

    traverse(node)
    return patterns


def main():
    print("Loading Chinese Wikipedia dataset from HuggingFace...")
    dataset = load_dataset("Jax-dan/zhwiki-latest", split="train", streaming=True)
    dataset = dataset.take(1000)  # Limit for testing

    print("Extracting and counting Chinese characters...")
    character_counter = Counter()

    # Process dataset
    for item in tqdm(dataset, desc="Processing articles"):
        text = item.get('text', '')
        characters = extract_chinese_characters(text)
        character_counter.update(characters)

    print(f"\nTotal unique characters found: {len(character_counter)}")
    print(f"Total character occurrences: {sum(character_counter.values())}")

    print("\nDecomposing characters and analyzing all subtree patterns...")
    pattern_counter = Counter()
    characters_processed = 0
    characters_with_ids = 0

    for char, freq in tqdm(character_counter.items(), desc="Analyzing characters"):
        characters_processed += 1

        # Get IDS for character
        ids = get_ids_for_character(char)
        if ids is None:
            continue

        characters_with_ids += 1

        try:
            # Parse IDS into tree
            tree = parse_ideographic_description_sequences(ids)

            # Find all subtree patterns
            patterns = find_all_subtree_patterns(tree)

            # Count patterns weighted by character frequency
            for pattern in patterns:
                pattern_counter[pattern] += freq
        except Exception:  # noqa: BLE001
            # Skip characters that fail to parse (various parsing errors possible from IDS data)
            pass

    print(f"\nCharacters processed: {characters_processed}")
    print(f"Characters with IDS: {characters_with_ids}")
    print(f"Unique subtree patterns found: {len(pattern_counter)}")

    # Print most common patterns
    print("\n" + "="*80)
    print("MOST COMMON SUBTREE PATTERNS (sorted by compression = price * frequency)")
    print("="*80)
    print(f"{'Rank':<6} {'Compression':<15} {'Frequency':<15} {'Pattern':<8} {'Character (if Exists)'}")
    print("-"*80)

    pricing = Counter({pattern: len(pattern) * count for pattern, count in pattern_counter.items()})

    for rank, (pattern, price) in enumerate(pricing.most_common(50), 1):
        freq = pattern_counter[pattern]
        pattern_str = "".join(pattern)
        print(f"{rank:<6} {price:<15,} {freq:<15,} {pattern_str:<8} {get_character_for_ids(pattern_str)}")

    # Calculate coverage
    total_pattern_frequency = sum(pattern_counter.values())
    total_char_frequency = sum(character_counter.values())
    coverage = (total_pattern_frequency / total_char_frequency) * 100 if total_char_frequency > 0 else 0

    print("\n" + "="*80)
    print(f"Total pattern occurrences: {total_pattern_frequency:,}")
    print(f"Coverage of all characters: {coverage:.2f}%")
    print("="*80)


if __name__ == "__main__":
    main()
