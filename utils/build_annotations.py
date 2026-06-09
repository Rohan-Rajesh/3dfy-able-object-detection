"""
Build a YTVIS-format annotation JSON from intermediate data.

Expected input layout:
  root/
    batch_001/
      output/
        step1_frames/{video}/frame_*.jpg
        step2_tracking/fwd_refined_masks/{video}/{video}_objects.json
        step2_tracking/fwd_refined_masks/{video}/{video}_rle.json
        step4.5_completeness/json/results.json
        vipe_poses/{video}/pose/images.npz        (required — videos without it are skipped)
        final_3d_assets/{video}_obj{idx}_accepted.glb   (optional)
        final_3d_assets/{video}_obj{idx}_rejected.glb   (optional)
    batch_002/
      output/
        ...

Positive gate (per object, in priority order):
  1. If final_3d_assets/ exists for the batch:
       include only objects with a matching _accepted.glb file.
  2. Otherwise fall back to results.json: include if is_suitable == YES.

Pose data (images.npz):
  - "data":  (N, 4, 4) float32 — camera transform matrix per pose frame
  - "inds":  (N,)      int64   — original video frame index for each pose
  Videos missing the npz file are silently skipped.
  Selected frames without a matching entry in "inds" get a null pose.

Subfolders not starting with "batch_" are ignored.
All batches are merged into a single output JSON.
"""

import json
import re
import argparse
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
import pycocotools.mask as mask_util

CATEGORY_RECONSTRUCTABLE = 1

_GLB_RE = re.compile(r"^(.+)_obj(\d+)_accepted\.glb$")


def rescale_rle(rle: dict, new_height: int, new_width: int) -> dict:
    orig_height, orig_width = rle["size"]
    if orig_height == new_height and orig_width == new_width:
        return rle
    mask = mask_util.decode(rle)
    resized = np.array(
        PILImage.fromarray(mask).resize((new_width, new_height), PILImage.NEAREST),
        dtype=np.uint8,
    )
    new_rle = mask_util.encode(np.asfortranarray(resized))
    new_rle["counts"] = new_rle["counts"].decode("utf-8")
    return new_rle


def load_glb_accepted(glb_root: Path) -> set[tuple[str, int]]:
    """Return {(video_name, obj_idx)} for every *_accepted.glb in glb_root."""
    accepted = set()
    for glb_file in glb_root.glob("*_accepted.glb"):
        m = _GLB_RE.match(glb_file.name)
        if m:
            accepted.add((m.group(1), int(m.group(2))))
    return accepted


def has_pose_data(poses_root: Path, seq_name: str) -> bool:
    return (poses_root / seq_name / "pose" / "images.npz").exists()


def load_pose_data(poses_root: Path, seq_name: str, total_frames: int) -> dict[int, list]:
    """Load per-frame pose matrices for a video.

    If the npz inds are sequential (0, 1, ..., N-1) — meaning VIPE used its own
    input-list indices rather than original frame numbers — the N poses are
    first mapped evenly across [0, total_frames-1].

    Every original frame is then assigned its nearest keyframe pose so there are
    no gaps (no None entries in the output).

    Returns {original_frame_idx: 4x4_matrix_as_nested_list} for all frames 0..total_frames-1.
    """
    d = np.load(poses_root / seq_name / "pose" / "images.npz")
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


def is_object_positive(
    seq_name: str,
    obj_idx: int,
    obj_comp: dict | None,
    glb_accepted: set[tuple[str, int]] | None,
) -> bool:
    """Return True if this object should be included as a positive annotation."""
    if glb_accepted is not None:
        return (seq_name, obj_idx) in glb_accepted
    # fallback: completeness gate
    if obj_comp is None:
        return False
    return obj_comp.get("is_suitable", "NO").strip().upper() == "YES"


def process_video(
    seq_name: str,
    frames_root: Path,
    tracking_root: Path,
    completeness_data: dict,
    glb_accepted: set[tuple[str, int]] | None,
    poses_root: Path,
    video_id: int,
    ann_id: int,
) -> tuple[dict | None, list[dict], int]:
    """Process one video. Returns (video_entry, annotation_entries, next_ann_id)."""
    frames_dir = frames_root / seq_name
    tracking_dir = tracking_root / seq_name

    obj_file = tracking_dir / f"{seq_name}_objects.json"
    rle_file = tracking_dir / f"{seq_name}_rle.json"

    if not obj_file.exists() or not rle_file.exists():
        print(f"  [SKIP] {seq_name}: missing tracking files")
        return None, [], ann_id

    # When using GLB gate we don't need completeness_data per-video, but we
    # still require the video to have been through the completeness pipeline
    # so its tracking data is trustworthy.
    if glb_accepted is None and seq_name not in completeness_data:
        print(f"  [SKIP] {seq_name}: not in completeness data")
        return None, [], ann_id

    if not has_pose_data(poses_root, seq_name):
        print(f"  [SKIP] {seq_name}: no vipe poses found")
        return None, [], ann_id

    frame_files = sorted(f.name for f in frames_dir.iterdir() if f.suffix == ".jpg")
    if not frame_files:
        print(f"  [SKIP] {seq_name}: no frames found")
        return None, [], ann_id

    with open(obj_file) as f:
        obj_data = json.load(f)
    with open(rle_file) as f:
        rle_data = json.load(f)

    total_frames = obj_data["total_frames"]
    pose_map = load_pose_data(poses_root, seq_name, total_frames)
    object_names = obj_data["objects"]
    video_completeness = completeness_data.get(seq_name, {})

    frame_skip = max(1, total_frames // 300)
    selected_indices = [i for i in range(total_frames) if i % frame_skip == 0]

    num_saved = len(frame_files)
    if len(selected_indices) != num_saved:
        print(
            f"  [WARN] {seq_name}: {len(selected_indices)} sampled indices "
            f"but {num_saved} saved frames — truncating to min"
        )
        n = min(len(selected_indices), num_saved)
        selected_indices = selected_indices[:n]
        frame_files = frame_files[:n]

    with PILImage.open(frames_dir / frame_files[0]) as img:
        width, height = img.size  # actual saved frame dimensions

    video_entry = {
        "id": video_id,
        "file_names": [f"{seq_name}/{fname}" for fname in frame_files],
        "height": height,
        "width": width,
        "length": len(frame_files),
        "poses": [pose_map.get(orig_idx) for orig_idx in selected_indices],
    }

    ann_entries = []
    n_reconstructable = n_skipped = 0

    for obj_idx, obj_name in enumerate(object_names):
        obj_comp = video_completeness.get(str(obj_idx))
        if not is_object_positive(seq_name, obj_idx, obj_comp, glb_accepted):
            n_skipped += 1
            continue

        n_reconstructable += 1
        segmentations, areas, bboxes = [], [], []
        for orig_idx in selected_indices:
            rle = rescale_rle(rle_data[orig_idx][obj_idx], height, width)
            area = float(mask_util.area(rle))
            if area == 0:
                segmentations.append(None)
                areas.append(None)
                bboxes.append(None)
            else:
                segmentations.append(rle)
                areas.append(area)
                bboxes.append(mask_util.toBbox(rle).tolist())

        ann_entries.append({
            "id": ann_id,
            "video_id": video_id,
            "category_id": CATEGORY_RECONSTRUCTABLE,
            "segmentations": segmentations,
            "areas": areas,
            "bboxes": bboxes,
            "iscrowd": 0,
        })
        ann_id += 1

    gate_source = "GLB" if glb_accepted is not None else "results.json"
    print(
        f"  {seq_name}: {len(frame_files)} frames, "
        f"{n_reconstructable} reconstructable [{gate_source}], "
        f"{n_skipped} skipped"
    )
    return video_entry, ann_entries, ann_id


def build_annotations(root: Path, output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)

    categories = [
        {"id": CATEGORY_RECONSTRUCTABLE, "name": "reconstructable", "supercategory": ""},
    ]

    videos = []
    annotations = []
    video_id = 0
    ann_id = 0

    batch_dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("batch_"))
    if not batch_dirs:
        print(f"[ERROR] No batch_* subdirectories found in {root}")
        return

    for batch_dir in batch_dirs:
        output_dir = batch_dir / "output"
        frames_root = output_dir / "step1_frames"
        tracking_root = output_dir / "step2_tracking" / "fwd_refined_masks"
        completeness_file = output_dir / "step4.5_completeness" / "json" / "results.json"
        glb_root = output_dir / "final_3d_assets"
        poses_root = output_dir / "vipe_poses"

        missing = [p for p in (frames_root, tracking_root, completeness_file) if not p.exists()]
        if missing:
            print(f"[SKIP] {batch_dir.name}: missing {[str(p) for p in missing]}")
            continue

        print(f"\n[{batch_dir.name}]")

        with open(completeness_file) as f:
            completeness_data = json.load(f)

        if glb_root.exists():
            glb_accepted = load_glb_accepted(glb_root)
            print(f"  Using GLB gate: {len(glb_accepted)} accepted objects found")
        else:
            glb_accepted = None
            print(f"  No final_3d_assets/ found, falling back to results.json")

        seq_names = sorted(d.name for d in frames_root.iterdir() if d.is_dir())
        for seq_name in seq_names:
            video_id += 1
            video_entry, ann_entries, ann_id = process_video(
                seq_name, frames_root, tracking_root,
                completeness_data, glb_accepted, poses_root, video_id, ann_id,
            )
            if video_entry is None:
                video_id -= 1
                continue
            videos.append(video_entry)
            annotations.extend(ann_entries)

    out_file = output_path / "all_annotations.json"
    dataset = {"videos": videos, "annotations": annotations, "categories": categories}
    with open(out_file, "w") as f:
        json.dump(dataset, f)

    print(
        f"\nSaved {len(videos)} videos, {len(annotations)} reconstructable annotations -> {out_file}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build YTVIS-format annotations from batched pipeline output")
    parser.add_argument("root", help="Root folder containing batch_* subdirectories")
    parser.add_argument("output", help="Folder to write all_annotations.json into")
    args = parser.parse_args()

    build_annotations(Path(args.root), Path(args.output))


if __name__ == "__main__":
    main()
