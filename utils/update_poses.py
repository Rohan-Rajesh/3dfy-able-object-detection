"""
Update pose data in an existing YTVIS-format annotation JSON using new VIPE NPZ files.

Usage:
    python utils/update_poses.py <json_file> <pose_dir> [--output <out_json>]

If --output is omitted, the input JSON is overwritten in place.

Pose directory structure:
    pose_dir/
        <video_id>/
            pose/
                images.npz

Frame indices are parsed from filenames (e.g. frame_000134.jpg → 134).
Videos with no matching NPZ file have their poses set to null and are warned.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_pose_data(pose_dir: Path, seq_name: str, total_frames: int) -> dict[int, list]:
    d = np.load(pose_dir / seq_name / "pose" / "images.npz")
    inds = d["inds"]
    data = d["data"]
    n = len(inds)

    if list(inds) == list(range(n)):
        keyframe_positions = [round(i * (total_frames - 1) / max(n - 1, 1)) for i in range(n)]
    else:
        keyframe_positions = [int(idx) for idx in inds]

    pose_list = data.tolist()
    return {
        f: pose_list[min(range(n), key=lambda k: abs(keyframe_positions[k] - f))]
        for f in range(total_frames)
    }


def frame_index_from_name(filename: str) -> int | None:
    """Parse frame index from a filename like 'seq/frame_000134.jpg' → 134."""
    stem = Path(filename).stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])
    return None


def update_poses(json_path: Path, pose_dir: Path, output_path: Path) -> None:
    with open(json_path) as f:
        data = json.load(f)

    videos = data.get("videos", [])
    n_updated = n_skipped = 0

    for video in videos:
        file_names = video.get("file_names", [])
        if not file_names:
            continue

        seq_name = Path(file_names[0]).parts[0]
        npz_path = pose_dir / seq_name / "pose" / "images.npz"

        if not npz_path.exists():
            print(f"  [SKIP] {seq_name}: {npz_path} not found")
            video["poses"] = [None] * len(file_names)
            n_skipped += 1
            continue

        # Parse selected frame indices from filenames; fall back to positional
        selected_indices = []
        for fname in file_names:
            idx = frame_index_from_name(fname)
            selected_indices.append(idx if idx is not None else len(selected_indices))

        # Use len(inds) as total_frames — VIPE was run on all frames so n == total
        d = np.load(npz_path)
        total_frames = len(d["inds"])

        pose_map = load_pose_data(pose_dir, seq_name, total_frames)
        video["poses"] = [pose_map.get(idx) for idx in selected_indices]

        print(f"  [OK]   {seq_name}: {len(selected_indices)} frames, {total_frames} poses in NPZ")
        n_updated += 1

    with open(output_path, "w") as f:
        json.dump(data, f)

    print(f"\nUpdated {n_updated} videos, skipped {n_skipped} → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Update pose data in an annotation JSON")
    parser.add_argument("json_file", help="Existing YTVIS annotation JSON")
    parser.add_argument("pose_dir", help="Pose directory: <pose_dir>/<video_id>/pose/images.npz")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: overwrite input)")
    args = parser.parse_args()

    json_path = Path(args.json_file)
    output_path = Path(args.output) if args.output else json_path
    update_poses(json_path, Path(args.pose_dir), output_path)


if __name__ == "__main__":
    main()
