"""
List video IDs and names from YTVIS-format annotation JSON files.

Usage:
    python3 utils/list_video_ids.py train.json valid.json
    python3 utils/list_video_ids.py train.json -o train_ids.txt

Each output line: <video_id>\t<video_name>
"""

import json
import argparse
from pathlib import Path


def extract_ids(json_path: Path) -> list[tuple[int, str]]:
    with open(json_path) as f:
        data = json.load(f)
    results = []
    for v in data["videos"]:
        # file_names is a list of "{seq_name}/frame_*.jpg" — extract folder name
        name = Path(v["file_names"][0]).parent.name if v["file_names"] else str(v["id"])
        results.append((v["id"], name))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="List video IDs from annotation JSON files")
    parser.add_argument("inputs", nargs="+", help="One or more annotation JSON files")
    parser.add_argument("-o", "--output", help="Write output to this file instead of stdout")
    args = parser.parse_args()

    lines = []
    for json_path in args.inputs:
        path = Path(json_path)
        entries = extract_ids(path)
        lines.append(f"# {path.name}  ({len(entries)} videos)")
        for vid_id, name in entries:
            lines.append(f"{vid_id}\t{name}")
        lines.append("")

    text = "\n".join(lines)
    if args.output:
        Path(args.output).write_text(text)
        print(f"Wrote {sum(1 for l in lines if l and not l.startswith('#'))} entries -> {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
