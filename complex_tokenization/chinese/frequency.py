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

from complex_tokenization.chinese.ideographic_description_sequences import (
    get_ids_for_character,
    parse_ideographic_description_sequences,
    IDSNode, get_character_for_ids
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


def find_near_leaf_nodes(node: IDSNode) -> list[tuple[str, ...]]:
    """
    Find all near-leaf nodes in the tree.

    A near-leaf node is a node whose children are all leaf nodes.
    Returns a list of tuples (node_value, child1_value, child2_value, ...)
    """
    near_leaf_patterns = []

    def traverse(node: IDSNode):
        if node.is_leaf():
            return

        # Check if all children are leaves
        if all(child.is_leaf() for child in node.children):
            # This is a near-leaf node
            pattern = tuple([node.value] + [child.value for child in node.children])
            near_leaf_patterns.append(pattern)

        # Continue traversing
        for child in node.children:
            traverse(child)

    traverse(node)
    return near_leaf_patterns


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

    print("\nDecomposing characters and analyzing near-leaf patterns...")
    near_leaf_counter = Counter()
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

            # Find near-leaf nodes
            patterns = find_near_leaf_nodes(tree)

            # Count patterns weighted by character frequency
            for pattern in patterns:
                near_leaf_counter[pattern] += freq
        except Exception as e:
            # Skip characters that fail to parse
            pass

    print(f"\nCharacters processed: {characters_processed}")
    print(f"Characters with IDS: {characters_with_ids}")
    print(f"Unique near-leaf patterns found: {len(near_leaf_counter)}")

    # Print most common patterns
    print("\n" + "="*80)
    print("MOST COMMON NEAR-LEAF PATTERNS")
    print("="*80)
    print(f"{'Rank':<6} {'Compression':<15} {'Frequency':<15} {'Pattern':<8} {'Character (if Exists)'}")
    print("-"*80)

    pricing = Counter({pattern: len(pattern) * count for pattern, count in near_leaf_counter.items()})

    for rank, (pattern, price) in enumerate(pricing.most_common(50), 1):
        freq = near_leaf_counter[pattern]
        pattern_str = "".join(pattern)
        print(f"{rank:<6} {price:<15,} {freq:<15,} {pattern_str:<8} {get_character_for_ids(pattern_str)}")

    # Calculate coverage
    total_pattern_frequency = sum(near_leaf_counter.values())
    total_char_frequency = sum(character_counter.values())
    coverage = (total_pattern_frequency / total_char_frequency) * 100 if total_char_frequency > 0 else 0

    print("\n" + "="*80)
    print(f"Total pattern occurrences: {total_pattern_frequency:,}")
    print(f"Coverage of all characters: {coverage:.2f}%")
    print("="*80)


if __name__ == "__main__":
    main()