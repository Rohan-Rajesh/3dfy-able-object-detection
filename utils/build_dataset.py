"""
Build a dataset folder compatible with build_annotations.py by selecting
videos according to a positive/negative ratio across one or more pipeline roots.

Usage:
    python3 utils/build_dataset.py \
        --roots /data/run1 /data/run2 \
        --max-videos 200 \
        --positive-ratio 0.8 \
        --output /data/dataset_v1 \
        [--ignore-json /data/test.json] \
        [--seed 42]

Output folder structure (symlink-based, ready for build_annotations.py):
    output/
      batch_r0_001/
        output/
          step1_frames/{video} -> symlink
          step2_tracking/fwd_refined_masks/{video} -> symlink
          step4.5_completeness/json/results.json -> symlink
          vipe_poses/{video} -> symlink
          final_3d_assets/ -> symlink (whole dir)
      batch_r1_001/
        ...

Positive gate (per video):
  - If final_3d_assets/ exists: positive = at least one {video}_obj*_accepted.glb
  - Otherwise: positive = at least one object with is_suitable==YES in results.json
"""

import argparse
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

_GLB_ACCEPTED_RE = re.compile(r"^(.+)_obj\d+_accepted\.glb$")


@dataclass
class VideoInfo:
    seq_name: str
    batch_key: str        # e.g. "batch_r0_001" — output dir name
    batch_output_dir: Path  # original batch's output/ dir
    is_positive: bool


def load_ignore_set(ignore_json: Path) -> set[str]:
    with open(ignore_json) as f:
        data = json.load(f)
    videos = data.get("videos", []) if isinstance(data, dict) else []
    return {Path(v["file_names"][0]).parent.name for v in videos if v.get("file_names")}


def is_positive_glb(seq_name: str, glb_root: Path) -> bool:
    return any(True for _ in glb_root.glob(f"{seq_name}_obj*_accepted.glb"))


def is_positive_completeness(seq_name: str, completeness_data: dict) -> bool:
    return any(
        obj.get("is_suitable", "NO").strip().upper() == "YES"
        for obj in completeness_data.get(seq_name, {}).values()
    )


def scan_root(root: Path, root_idx: int) -> list[VideoInfo]:
    videos = []
    batch_dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("batch_"))
    if not batch_dirs:
        print(f"  [WARN] No batch_* dirs found in {root}")
        return videos

    for batch_dir in batch_dirs:
        batch_suffix = batch_dir.name[len("batch_"):]
        batch_key = f"batch_r{root_idx}_{batch_suffix}"
        output_dir = batch_dir / "output"
        frames_root = output_dir / "step1_frames"
        tracking_root = output_dir / "step2_tracking" / "fwd_refined_masks"
        completeness_file = output_dir / "step4.5_completeness" / "json" / "results.json"
        glb_root = output_dir / "final_3d_assets"
        poses_root = output_dir / "vipe_poses"

        if not frames_root.exists() or not tracking_root.exists():
            continue

        completeness_data = {}
        if completeness_file.exists():
            with open(completeness_file) as f:
                completeness_data = json.load(f)

        use_glb = glb_root.exists()

        for video_dir in sorted(frames_root.iterdir()):
            if not video_dir.is_dir():
                continue
            seq_name = video_dir.name

            obj_file = tracking_root / seq_name / f"{seq_name}_objects.json"
            rle_file = tracking_root / seq_name / f"{seq_name}_rle.json"
            if not obj_file.exists() or not rle_file.exists():
                continue

            pose_file = poses_root / seq_name / "pose" / "images.npz"
            if not pose_file.exists():
                continue

            positive = (
                is_positive_glb(seq_name, glb_root)
                if use_glb
                else is_positive_completeness(seq_name, completeness_data)
            )

            videos.append(VideoInfo(
                seq_name=seq_name,
                batch_key=batch_key,
                batch_output_dir=output_dir,
                is_positive=positive,
            ))

    return videos


def symlink(src: Path, dst: Path) -> None:
    if not dst.exists() and not dst.is_symlink():
        dst.symlink_to(src.resolve())


def create_output_batch(batch_key: str, videos: list[VideoInfo], output_root: Path) -> None:
    src = videos[0].batch_output_dir
    dst = output_root / batch_key / "output"

    frames_dst = dst / "step1_frames"
    tracking_dst = dst / "step2_tracking" / "fwd_refined_masks"
    frames_dst.mkdir(parents=True, exist_ok=True)
    tracking_dst.mkdir(parents=True, exist_ok=True)

    poses_dst = dst / "vipe_poses"
    poses_dst.mkdir(parents=True, exist_ok=True)

    for v in videos:
        symlink(src / "step1_frames" / v.seq_name, frames_dst / v.seq_name)
        symlink(src / "step2_tracking" / "fwd_refined_masks" / v.seq_name, tracking_dst / v.seq_name)
        symlink(src / "vipe_poses" / v.seq_name, poses_dst / v.seq_name)

    # Completeness file (batch-level)
    comp_src = src / "step4.5_completeness" / "json" / "results.json"
    if comp_src.exists():
        comp_dst_dir = dst / "step4.5_completeness" / "json"
        comp_dst_dir.mkdir(parents=True, exist_ok=True)
        symlink(comp_src, comp_dst_dir / "results.json")

    # GLB dir (batch-level, whole dir symlinked — build_annotations filters per video)
    glb_src = src / "final_3d_assets"
    if glb_src.exists():
        symlink(glb_src, dst / "final_3d_assets")


def build_dataset(
    roots: list[Path],
    output: Path,
    ignore_json: Path | None,
    seed: int,
    max_videos: int | None = None,
    positive_ratio: float | None = None,
) -> None:
    rng = random.Random(seed)

    ignore_set: set[str] = set()
    if ignore_json is not None:
        ignore_set = load_ignore_set(ignore_json)
        print(f"Ignore list: {len(ignore_set)} sequences from {ignore_json.name}")

    print("\nScanning roots...")
    all_videos: list[VideoInfo] = []
    for i, root in enumerate(roots):
        found = scan_root(root, i)
        n_pos = sum(v.is_positive for v in found)
        print(f"  root {i} ({root.name}): {len(found)} videos ({n_pos} positive, {len(found)-n_pos} negative)")
        all_videos.extend(found)

    all_videos = [v for v in all_videos if v.seq_name not in ignore_set]
    positives = [v for v in all_videos if v.is_positive]
    negatives = [v for v in all_videos if not v.is_positive]
    print(f"\nAfter filtering: {len(positives)} positive, {len(negatives)} negative available")

    if positive_ratio is None and max_videos is None:
        # Take everything at natural ratio
        selected = list(all_videos)
        rng.shuffle(selected)
        n_positive = len(positives)
        n_negative = len(negatives)
    else:
        ratio = positive_ratio if positive_ratio is not None else len(positives) / max(len(all_videos), 1)
        total = max_videos if max_videos is not None else len(all_videos)

        n_positive = round(total * ratio)
        n_negative = total - n_positive

        if len(positives) < n_positive:
            print(f"[WARN] Only {len(positives)} positives available, wanted {n_positive} — using all")
            n_positive = len(positives)
            n_negative = min(n_negative, total - n_positive)

        if len(negatives) < n_negative:
            print(f"[WARN] Only {len(negatives)} negatives available, wanted {n_negative} — using all")
            n_negative = len(negatives)

        selected = rng.sample(positives, n_positive) + rng.sample(negatives, n_negative)
        rng.shuffle(selected)

    print(f"Selected: {n_positive} positive + {n_negative} negative = {len(selected)} total")

    by_batch: dict[str, list[VideoInfo]] = defaultdict(list)
    for v in selected:
        by_batch[v.batch_key].append(v)

    output.mkdir(parents=True, exist_ok=True)
    print("\nBuilding output structure...")
    for batch_key, batch_videos in sorted(by_batch.items()):
        n_pos = sum(v.is_positive for v in batch_videos)
        print(f"  {batch_key}: {len(batch_videos)} videos ({n_pos} pos, {len(batch_videos)-n_pos} neg)")
        create_output_batch(batch_key, batch_videos, output)

    print(f"\nDone -> {output}")
    print(f"Next: python3 utils/build_annotations.py {output} <annotations_dir>")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a dataset folder (symlink-based) compatible with build_annotations.py"
    )
    parser.add_argument("--roots", nargs="+", required=True,
                        help="Pipeline root folders, each containing batch_xxx subdirs")
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Total number of videos to include (default: all available)")
    parser.add_argument("--positive-ratio", type=float, default=None,
                        help="Fraction of videos that should be positive (default: natural ratio)")
    parser.add_argument("--output", required=True,
                        help="Output dataset folder (structured for build_annotations.py)")
    parser.add_argument("--ignore-json", default=None,
                        help="YTVIS-format JSON with sequences to exclude (e.g. test set)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    if args.positive_ratio is not None and not 0.0 <= args.positive_ratio <= 1.0:
        parser.error("--positive-ratio must be between 0.0 and 1.0")

    build_dataset(
        roots=[Path(r) for r in args.roots],
        output=Path(args.output),
        ignore_json=Path(args.ignore_json) if args.ignore_json else None,
        seed=args.seed,
        max_videos=args.max_videos,
        positive_ratio=args.positive_ratio,
    )


if __name__ == "__main__":
    main()

