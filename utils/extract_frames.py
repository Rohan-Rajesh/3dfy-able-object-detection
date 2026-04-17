"""
Extract frames from mp4 videos in intermediate_data/molmospaces_100vid_export/
and save them to intermediate_data/step1_frames/.

For each video:
  - Sample down to at most 300 frames (evenly spaced)
  - Resize so the longest dimension is at most 720px (aspect ratio preserved)
  - Keep every frame_skip-th frame (frame_idx % frame_skip == 0)
"""

import os
import cv2
import argparse
from pathlib import Path


MAX_FRAMES = 300
MAX_DIM = 720


def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return total_frames, width, height, fps


def compute_output_size(width, height):
    if max(width, height) > MAX_DIM:
        if width > height:
            new_width = MAX_DIM
            new_height = int(height * MAX_DIM / width)
        else:
            new_height = MAX_DIM
            new_width = int(width * MAX_DIM / height)
    else:
        new_width, new_height = width, height
    return new_width, new_height


def extract_frames(video_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    total_frames, width, height, fps = get_video_info(video_path)
    frame_skip = max(1, total_frames // MAX_FRAMES)
    new_width, new_height = compute_output_size(width, height)

    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            if new_width != width or new_height != height:
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            out_path = output_dir / f"{saved:06d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1

        frame_idx += 1

    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser(description="Extract frames from videos")
    parser.add_argument(
        "--input-dir",
        default="../intermediate_data/molmospaces_100vid_export",
        help="Directory containing per-video subfolders with mp4 files",
    )
    parser.add_argument(
        "--output-dir",
        default="../intermediate_data/step1_frames",
        help="Output directory for extracted frames",
    )
    args = parser.parse_args()

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

    video_dirs = sorted(p for p in input_root.iterdir() if p.is_dir())

    print(f"Found {len(video_dirs)} video directories")
    print(f"Output: {output_root}\n")

    for video_dir in video_dirs:
        video_path = video_dir / f"{video_dir.name}.mp4"
        if not video_path.exists():
            print(f"  [SKIP] {video_dir.name}: mp4 not found at {video_path}")
            continue

        out_dir = output_root / video_dir.name
        n = extract_frames(video_path, out_dir)
        print(f"  {video_dir.name}: {n} frames saved")

    print("\nDone.")


if __name__ == "__main__":
    main()
