"""
Build a YTVIS-format annotation JSON from intermediate data.

Reads:
  - intermediate_data/step1_frames/{video}/     frame images (determines file_names)
  - intermediate_data/step2_tracking/{video}/*_objects.json   object names, total_frames
  - intermediate_data/step2_tracking/{video}/*_rle.json       per-frame RLE masks

Outputs:
  - intermediate_data/annotations/all_annotations.json

Categories are derived globally: all unique object names across every video are
collected, sorted, and assigned contiguous integer IDs starting at 1.

Frame alignment:  extract_frames.py saves every frame_skip-th original frame,
where frame_skip = max(1, total_frames // 300).  This script replicates that
sampling to select the matching rows from rle.json.
"""

import json
import argparse
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
import pycocotools.mask as mask_util

MAX_DIM = 720  # must match extract_frames.py


def resized_dims(orig_height: int, orig_width: int) -> tuple[int, int]:
    """Return (new_height, new_width) applying the same resize logic as extract_frames.py."""
    if max(orig_width, orig_height) > MAX_DIM:
        if orig_width > orig_height:
            new_width = MAX_DIM
            new_height = int(orig_height * MAX_DIM / orig_width)
        else:
            new_height = MAX_DIM
            new_width = int(orig_width * MAX_DIM / orig_height)
    else:
        new_height, new_width = orig_height, orig_width
    return new_height, new_width


def rescale_rle(rle: dict, new_height: int, new_width: int) -> dict:
    """Decode RLE mask, resize to (new_height, new_width), re-encode."""
    orig_height, orig_width = rle["size"]
    if orig_height == new_height and orig_width == new_width:
        return rle
    mask = mask_util.decode(rle)  # uint8 array (orig_height, orig_width)
    resized = np.array(
        PILImage.fromarray(mask).resize((new_width, new_height), PILImage.NEAREST),
        dtype=np.uint8,
    )
    new_rle = mask_util.encode(np.asfortranarray(resized))
    new_rle["counts"] = new_rle["counts"].decode("utf-8")
    return new_rle


def build_category_map(tracking_root: Path) -> dict[str, int]:
    """Return {object_name: category_id}. All objects map to the single 'object' class (id=1)."""
    names: set[str] = set()
    for seq_dir in tracking_root.iterdir():
        if not seq_dir.is_dir():
            continue
        obj_file = seq_dir / f"{seq_dir.name}_objects.json"
        if not obj_file.exists():
            continue
        with open(obj_file) as f:
            names.update(json.load(f)["objects"])
    return {name: 1 for name in names}


def build_annotations(frames_root: Path, tracking_root: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cat_map = build_category_map(tracking_root)
    categories = [{"id": 1, "name": "object", "supercategory": ""}]

    videos = []
    annotations = []
    video_id = 0
    ann_id = 0

    seq_names = sorted(d.name for d in frames_root.iterdir() if d.is_dir())

    for seq_name in seq_names:
        frames_dir = frames_root / seq_name
        tracking_dir = tracking_root / seq_name

        obj_file = tracking_dir / f"{seq_name}_objects.json"
        rle_file = tracking_dir / f"{seq_name}_rle.json"

        if not obj_file.exists() or not rle_file.exists():
            print(f"  [SKIP] {seq_name}: missing tracking files")
            continue

        frame_files = sorted(f.name for f in frames_dir.iterdir() if f.suffix == ".jpg")
        if not frame_files:
            print(f"  [SKIP] {seq_name}: no frames found in step1_frames")
            continue

        with open(obj_file) as f:
            obj_data = json.load(f)
        with open(rle_file) as f:
            rle_data = json.load(f)  # rle_data[orig_frame_idx][obj_idx] = {size, counts}

        total_frames = obj_data["total_frames"]
        object_names = obj_data["objects"]

        # Replicate the sampling logic from extract_frames.py
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

        # Compute actual saved-frame dimensions (extract_frames.py resizes the video)
        orig_height, orig_width = rle_data[0][0]["size"]
        height, width = resized_dims(orig_height, orig_width)

        video_id += 1
        videos.append(
            {
                "id": video_id,
                "file_names": [f"{seq_name}/{fname}" for fname in frame_files],
                "height": height,
                "width": width,
                "length": len(frame_files),
            }
        )

        for obj_idx, obj_name in enumerate(object_names):
            segmentations = []
            areas = []
            bboxes = []

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
                    bboxes.append(mask_util.toBbox(rle).tolist())  # [x, y, w, h]

            ann_id += 1
            annotations.append(
                {
                    "id": ann_id,
                    "video_id": video_id,
                    "category_id": cat_map[obj_name],
                    "segmentations": segmentations,
                    "areas": areas,
                    "bboxes": bboxes,
                    "iscrowd": 0,
                }
            )

        print(f"  {seq_name}: {len(frame_files)} frames, {len(object_names)} objects")

    dataset = {"videos": videos, "annotations": annotations, "categories": categories}
    with open(output_path, "w") as f:
        json.dump(dataset, f)

    print(
        f"\nSaved {len(videos)} videos, {len(annotations)} annotations, "
        f"{len(categories)} categories -> {output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build YTVIS-format annotations")
    parser.add_argument("--frames-dir", default="../intermediate_data/step1_frames")
    parser.add_argument("--tracking-dir", default="../intermediate_data/step2_tracking")
    parser.add_argument(
        "--output", default="../intermediate_data/annotations/all_annotations.json"
    )
    args = parser.parse_args()

    build_annotations(Path(args.frames_dir), Path(args.tracking_dir), Path(args.output))


if __name__ == "__main__":
    main()
