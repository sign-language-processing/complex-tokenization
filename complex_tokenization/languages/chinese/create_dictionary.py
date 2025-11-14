import json
import subprocess
from pathlib import Path


def clone_repo_if_needed(repo_url, repo_name):
    """Clone repository if it doesn't exist."""
    repo_dir = Path(__file__).parent / repo_name
    if not repo_dir.exists():
        print(f"Cloning {repo_name} repository to {repo_dir}...")
        subprocess.run(
            ["git", "clone", repo_url, str(repo_dir)],
            check=True
        )
    return repo_dir


def extract_ids(files):
    """Extract ids.txt and ids-ext-cdef.txt from cjkvi-ids."""
    dictionary = {}

    for file_path in files:
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping...")
            continue

        print(f"Processing {file_path.name}...")
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    key = parts[1].strip()
                    value = parts[2]
                    if "[" in value:
                        value = value[:value.index("[")]
                    value = value.strip()
                    if key != value:
                        dictionary[key] = value

    return dictionary


def load_canonicalization_rules(canonicalize_path):
    """Load canonicalization rules from canonicalize.txt."""
    rules = {}

    if not canonicalize_path.exists():
        print(f"Warning: {canonicalize_path} not found, skipping canonicalization...")
        return rules

    with open(canonicalize_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split('\t')
            if len(parts) < 3:
                continue

            rule_type = parts[0]
            source = parts[1]
            target = parts[2].replace("~", "")  # Remove any '~' characters

            # Only apply certain rule types for normalization
            if rule_type in ['identical', 'variant', 'print', 'preferred']:
                rules[source] = target

    return rules


def canonicalize_dictionary(dictionary, canonicalization_rules):
    """Canonicalize dictionary values using canonicalization rules."""
    if not canonicalization_rules:
        return dictionary

    print("Canonicalizing dictionary values...")
    canonicalized = {}

    for key, value in dictionary.items():
        # Apply canonicalization rules to each character in the value
        canonical_value = ""
        for char in value:
            canonical_value += canonicalization_rules.get(char, char)
        if key != canonical_value:
            canonicalized[key] = canonical_value

    return canonicalized


def expand_ids(value, dictionary):
    expanded_value = ""
    for char in value:
        if char in dictionary:
            dictionary[char] = expand_ids(dictionary[char], dictionary)
            expanded_value += dictionary[char]
        else:
            expanded_value += char
    return expanded_value


def expand_dictionary(dictionary):
    """Expand dictionary values by replacing characters that exist as keys."""
    print("Expanding dictionary values...")
    expanded = {}

    for key, value in dictionary.items():
        if "{" in value:
            continue

        expanded[key] = expand_ids(value, dictionary)

    return expanded


def save_dictionary(dictionary):
    """Save the dictionary to dictionary.json."""
    output_path = Path(__file__).parent / "dictionary.json"
    print(f"Saving dictionary to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(dictionary)} entries to {output_path}")


def main():
    # Clone both repositories
    hfhchan_repo = clone_repo_if_needed("https://github.com/hfhchan/ids.git", "hfhchan-ids")
    cjkvi_repo = clone_repo_if_needed("https://github.com/cjkvi/cjkvi-ids.git", "cjkvi-ids")

    # Load canonicalization rules
    canonicalization_rules = load_canonicalization_rules(hfhchan_repo / "canonicalize.txt")

    # Extract from hfhchan first, then cjkvi (cjkvi overwrites)
    dictionary = extract_ids([
        hfhchan_repo / "release" / "ids-20240112.txt",
        cjkvi_repo / "ids.txt",
        cjkvi_repo / "ids-ext-cdef.txt"
    ])

    # Canonicalize, then expand
    dictionary = canonicalize_dictionary(dictionary, canonicalization_rules)
    dictionary = expand_dictionary(dictionary)
    save_dictionary(dictionary)


if __name__ == "__main__":
    main()
