"""
Symlink batch_* directories from a source root into a destination root.

Usage:
    python3 utils/merge_batch_roots.py <source_root> <dest_root>

Example:
    python3 utils/merge_batch_roots.py /data/molmospaces_v2 /data/molmospaces

For each batch_* directory in source_root, creates a symlink inside dest_root.
If a name collision occurs (directory or symlink already exists), the entry is
skipped with a warning — nothing in dest_root is modified.
"""

import argparse
from pathlib import Path


def merge_batch_roots(source: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)

    batch_dirs = sorted(d for d in source.iterdir() if d.is_dir() and d.name.startswith("batch_"))
    if not batch_dirs:
        print(f"[ERROR] No batch_* subdirectories found in {source}")
        return

    n_linked = n_skipped = 0
    for batch_dir in batch_dirs:
        link = dest / batch_dir.name
        if link.exists() or link.is_symlink():
            print(f"  [SKIP] {batch_dir.name}: already exists in {dest} — not modified")
            n_skipped += 1
            continue
        link.symlink_to(batch_dir.resolve())
        print(f"  [LINK] {batch_dir.name} -> {batch_dir.resolve()}")
        n_linked += 1

    print(f"\nDone: {n_linked} symlinks created, {n_skipped} skipped")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Symlink batch_* dirs from source root into dest root"
    )
    parser.add_argument("source", help="Folder whose batch_* dirs will be symlinked")
    parser.add_argument("dest", help="Folder to receive the symlinks (existing contents untouched)")
    args = parser.parse_args()

    merge_batch_roots(Path(args.source), Path(args.dest))


if __name__ == "__main__":
    main()
