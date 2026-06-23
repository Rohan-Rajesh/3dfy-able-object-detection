"""
Temporary script: find videos where VIPE pose count doesn't match frame count,
and symlink those frame directories into an output folder for reprocessing.

Usage:
    python utils/find_pose_mismatch.py <json_file> <frames_dir> <pose_dir> <output_dir>

Output layout:
    output_dir/
        ark_41125231_clip1 -> <frames_dir>/ark_41125231_clip1   (symlink)
        ...
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", help="YTVIS annotation JSON")
    parser.add_argument("frames_dir", help="Root folder containing per-video frame subdirectories")
    parser.add_argument("pose_dir", help="Root folder containing per-video pose NPZ files")
    parser.add_argument("output_dir", help="Output folder to symlink mismatched video directories into")
    args = parser.parse_args()

    json_path = Path(args.json_file)
    frames_root = Path(args.frames_dir)
    pose_root = Path(args.pose_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    with open(json_path) as f:
        data = json.load(f)

    videos = data.get("videos", [])
    n_match = n_mismatch = n_missing = 0

    for video in videos:
        file_names = video.get("file_names", [])
        if not file_names:
            continue

        seq_name = Path(file_names[0]).parts[0]
        frame_dir = frames_root / seq_name
        npz_path = pose_root / seq_name / "pose" / "images.npz"

        n_frames = len(list(frame_dir.glob("*.jpg"))) if frame_dir.is_dir() else 0

        if not npz_path.exists():
            print(f"  [MISSING POSE] {seq_name}: {n_frames} frames, no NPZ")
            n_missing += 1
            _symlink(frame_dir, output_root / seq_name)
            continue

        d = np.load(npz_path)
        n_poses = len(d["inds"])

        if n_poses != n_frames:
            print(f"  [MISMATCH] {seq_name}: {n_frames} frames vs {n_poses} poses")
            n_mismatch += 1
            _symlink(frame_dir, output_root / seq_name)
        else:
            print(f"  [OK]       {seq_name}: {n_frames} frames == {n_poses} poses")
            n_match += 1

    print(f"\nMatch: {n_match}  |  Mismatch: {n_mismatch}  |  Missing pose: {n_missing}")
    print(f"Symlinked {n_mismatch + n_missing} directories into {output_root}")


def _symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


if __name__ == "__main__":
    main()
