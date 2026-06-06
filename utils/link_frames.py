"""
Create a flat symlink directory for all video frame folders across batches.

Usage:
    python3 utils/link_frames.py <root> <output_dir>

Example:
    python3 utils/link_frames.py /data/molmospaces /data/all_frames

Scans root/batch_*/output/step1_frames/ and creates one symlink per video
subfolder inside output_dir. Skips duplicates (same video name in multiple
batches) with a warning.
"""

import argparse
import os
from pathlib import Path


def link_frames(root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("batch_"))
    if not batch_dirs:
        print(f"[ERROR] No batch_* subdirectories found in {root}")
        return

    n_linked = 0
    n_skipped = 0

    for batch_dir in batch_dirs:
        frames_root = batch_dir / "output" / "step1_frames"
        if not frames_root.exists():
            print(f"[SKIP] {batch_dir.name}: no step1_frames directory")
            continue

        video_dirs = sorted(d for d in frames_root.iterdir() if d.is_dir())
        for video_dir in video_dirs:
            link = output_dir / video_dir.name
            if link.exists() or link.is_symlink():
                print(f"  [SKIP] {video_dir.name}: already exists in output (from another batch?)")
                n_skipped += 1
                continue
            link.symlink_to(video_dir.resolve())
            n_linked += 1

    print(f"\nDone: {n_linked} symlinks created, {n_skipped} skipped -> {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Symlink video frame folders into a flat directory")
    parser.add_argument("root", help="Root folder containing batch_* subdirectories")
    parser.add_argument("output_dir", help="Flat output directory to populate with symlinks")
    args = parser.parse_args()

    link_frames(Path(args.root), Path(args.output_dir))


if __name__ == "__main__":
    main()
