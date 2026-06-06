"""
Split all_annotations.json into train/valid subsets.

Videos are shuffled with a fixed random seed and split by ratio (default 80/20).
Annotations follow their video into the correct split.  Categories are copied
unchanged into both files.

Reads:  <output_dir>/all_annotations.json
Writes: <output_dir>/train.json
        <output_dir>/valid.json
"""

import json
import random
import argparse
from pathlib import Path


def split_dataset(
    input_path: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> None:
    with open(input_path) as f:
        dataset = json.load(f)

    videos = dataset["videos"]
    annotations = dataset["annotations"]
    categories = dataset["categories"]

    video_ids = [v["id"] for v in videos]
    rng = random.Random(seed)
    rng.shuffle(video_ids)

    split_point = int(len(video_ids) * train_ratio)
    train_ids = set(video_ids[:split_point])
    val_ids = set(video_ids[split_point:])

    # Build a lookup for fast annotation filtering
    ann_by_video: dict[int, list] = {}
    for ann in annotations:
        ann_by_video.setdefault(ann["video_id"], []).append(ann)

    for split_name, id_set in [("train", train_ids), ("valid", val_ids)]:
        split_videos = [v for v in videos if v["id"] in id_set]
        split_anns = [ann for vid_id in id_set for ann in ann_by_video.get(vid_id, [])]

        split_data = {
            "videos": split_videos,
            "annotations": split_anns,
            "categories": categories,
        }

        out_path = output_dir / f"{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(split_data, f)

        print(f"  {split_name}: {len(split_videos)} videos, {len(split_anns)} annotations -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/valid split for YTVIS annotations")
    parser.add_argument("input", help="Path to all_annotations.json")
    parser.add_argument("output_dir", help="Folder to write train.json and valid.json into")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_dataset(Path(args.input), output_dir, args.train_ratio, args.seed)


if __name__ == "__main__":
    main()
