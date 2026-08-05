"""
Copy all video frame folders across batches into a flat output directory,
optionally resizing frames to reduce disk usage.

Usage:
    python3 utils/link_frames.py <root> <output_dir> [--size 512] [--quality 85]

Example:
    python3 utils/link_frames.py /data/molmospaces /data/all_frames --size 512

Scans root/batch_*/output/step1_frames/ and copies each video subfolder
into output_dir. With --size, frames are resized (longest edge) during copy.
Skips already-complete videos; recopies partial ones (resume support).
"""

import argparse
import shutil
from pathlib import Path

from PIL import Image


def copy_video(src: Path, dst: Path, size: int | None, quality: int) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for frame in sorted(src.iterdir()):
        if not frame.is_file():
            continue
        out = dst / frame.name
        if size is not None:
            img = Image.open(frame)
            img.thumbnail((size, size), Image.LANCZOS)
            img.save(out, quality=quality, optimize=True)
        else:
            shutil.copy2(frame, out)


def link_frames(root: Path, output_dir: Path, size: int | None, quality: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("batch_"))
    if not batch_dirs:
        print(f"[ERROR] No batch_* subdirectories found in {root}")
        return

    jobs = []
    n_complete = 0
    for batch_dir in batch_dirs:
        frames_root = batch_dir / "output" / "step1_frames"
        if not frames_root.exists():
            print(f"[SKIP] {batch_dir.name}: no step1_frames directory")
            continue
        for video_dir in sorted(d for d in frames_root.iterdir() if d.is_dir()):
            dst = output_dir / video_dir.name
            if dst.exists():
                src_count = sum(1 for f in video_dir.iterdir() if f.is_file())
                dst_count = sum(1 for f in dst.iterdir() if f.is_file())
                if src_count == dst_count:
                    n_complete += 1
                    continue
                print(f"  [RESUME] {video_dir.name}: incomplete ({dst_count}/{src_count} files), recopying")
                shutil.rmtree(dst)
            jobs.append((video_dir, dst))

    if n_complete:
        print(f"Resuming: {n_complete} videos already complete, {len(jobs)} remaining")

    if not jobs:
        print("Nothing to copy.")
        return

    if size:
        print(f"Resizing longest edge to {size}px, quality={quality}")
    n_copied = 0
    total = len(jobs)
    for src, dst in jobs:
        copy_video(src, dst, size, quality)
        n_copied += 1
        remaining = total - n_copied
        print(f"  [DONE] {src.name} ({remaining} videos remaining)")

    print(f"\nDone: {n_copied} copied, {n_complete} already complete -> {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy video frame folders into a flat directory")
    parser.add_argument("root", help="Root folder containing batch_* subdirectories")
    parser.add_argument("output_dir", help="Flat output directory to populate with copies")
    parser.add_argument(
        "--size", type=int, default=None,
        help="Resize frames so the longest edge is this many pixels (e.g. 256 or 512). "
             "Omit to copy at original resolution.",
    )
    parser.add_argument(
        "--quality", type=int, default=85,
        help="JPEG quality for resized output, 1-95 (default: 85)",
    )
    args = parser.parse_args()

    link_frames(Path(args.root), Path(args.output_dir), args.size, args.quality)


if __name__ == "__main__":
    main()
