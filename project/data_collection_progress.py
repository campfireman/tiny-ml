import re
from os import path
from pathlib import Path
from collections import defaultdict


TARGET_COUNT = 600


def main():
    labels = defaultdict(int)
    total = 0
    for file in Path("data/").iterdir():
        if not file.is_file():
            continue
        m = re.search(
            r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-(\w+)", file.name)
        if not m:
            continue
        label = m.group(1)
        labels[label] += 1

    print("=== PROGRESS ===\n")
    for label, count in sorted(list(labels.items()), key=lambda x: x[1]):
        print(f"{label}: {count} ({count / TARGET_COUNT:.0%})")
        total += count
    print()
    print(f"Total: {total}")


if __name__ == "__main__":
    main()
