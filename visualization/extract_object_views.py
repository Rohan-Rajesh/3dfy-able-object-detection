#!/usr/bin/env python3
"""
Run inference on a video and save each tracked object's views on a white background.

Usage:
    python visualization/extract_object_views.py \
        --frames-dir intermediate_data/step1_frames/full_fast_FloorPlan203_physics \
        --output-dir object_views \
        --config-file configs/custom/videomt_custom_ViTL.yaml \
        --opts MODEL.WEIGHTS output_videomt_custom_ViTL/model_final.pth
"""

import argparse
import glob
import math
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config

from videomt import add_videomt_config
from predictor import VisualizationDemo_windows


def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_videomt_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def apply_mask_white_bg(frame_bgr, mask):
    """Return a crop of the object on a white background, tightly cropped to the bounding box."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = np.ones_like(frame_rgb) * 255  # white canvas
    result[mask] = frame_rgb[mask]

    # tight crop to bounding box
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    padding = 10
    r0 = max(0, r0 - padding)
    r1 = min(frame_rgb.shape[0], r1 + padding)
    c0 = max(0, c0 - padding)
    c1 = min(frame_rgb.shape[1], c1 + padding)
    return result[r0:r1, c0:c1]


def make_grid(crops, max_cols=5):
    """Arrange a list of cropped images into a grid on a white background."""
    if not crops:
        return None
    target_h = 200
    resized = []
    for crop in crops:
        h, w = crop.shape[:2]
        scale = target_h / h
        new_w = max(1, int(w * scale))
        resized.append(cv2.resize(crop, (new_w, target_h)))

    n_cols = min(max_cols, len(resized))
    n_rows = math.ceil(len(resized) / n_cols)
    col_w = max(r.shape[1] for r in resized)

    grid = np.ones((n_rows * target_h, n_cols * col_w, 3), dtype=np.uint8) * 255
    for i, img in enumerate(resized):
        row, col = divmod(i, n_cols)
        y, x = row * target_h, col * col_w
        grid[y:y + target_h, x:x + img.shape[1]] = img

    return grid


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", required=True, help="Directory of input JPEG frames")
    parser.add_argument("--output-dir", required=True, help="Directory to save per-object grids")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    parser.add_argument("--windows-size", type=int, default=20)
    parser.add_argument("--max-views", type=int, default=0,
                        help="Max frames to save per object (evenly sampled); 0 = all frames")
    parser.add_argument("--min-views", type=int, default=5,
                        help="Minimum frames an object must appear in to be saved")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Only save the top N instances by frame count")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=[])
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = setup_cfg(args)
    demo = VisualizationDemo_windows(cfg)

    frame_paths = sorted(glob.glob(os.path.join(args.frames_dir, "*.jpg")))
    assert frame_paths, f"No JPG frames found in {args.frames_dir}"
    print(f"Found {len(frame_paths)} frames")

    windows_size = args.windows_size if args.windows_size != -1 else len(frame_paths)

    # instance_id -> list of (frame_bgr, mask_2d, global_frame_idx)
    instance_views: dict[int, list] = {}

    vid_frames, frame_buf, frame_indices = [], [], []
    for i, path in enumerate(frame_paths):
        img = read_image(path, format="BGR")
        vid_frames.append(img)
        frame_buf.append(img)
        frame_indices.append(i)

        if len(vid_frames) == windows_size or i == len(frame_paths) - 1:
            with torch.amp.autocast(device_type="cuda"):
                keep = i >= windows_size
                predictions, _ = demo.run_on_video(vid_frames, keep=keep)

            scores = predictions.get("pred_scores", [])
            masks = predictions.get("pred_masks", [])   # list of (T, H, W) bool tensors
            ids = predictions.get("pred_ids", [])

            for score, mask_seq, inst_id in zip(scores, masks, ids):
                if score < args.confidence_threshold:
                    continue
                if inst_id not in instance_views:
                    instance_views[inst_id] = []
                # mask_seq shape: (T, H, W)
                mask_np = mask_seq.cpu().numpy() if hasattr(mask_seq, "cpu") else np.array(mask_seq)
                for t, (frame, fidx) in enumerate(zip(frame_buf, frame_indices)):
                    if t < mask_np.shape[0] and mask_np[t].any():
                        instance_views[inst_id].append((frame.copy(), mask_np[t], fidx))

            del vid_frames, frame_buf, frame_indices
            vid_frames, frame_buf, frame_indices = [], [], []

    print(f"Found {len(instance_views)} tracked instances before filtering")

    # --- Filter 1: drop instances that appear in too few frames ---
    instance_views = {
        iid: views for iid, views in instance_views.items()
        if len(views) >= args.min_views
    }
    print(f"  After min-views={args.min_views} filter: {len(instance_views)} instances")

    # --- Filter 2: keep only top-N by frame count ---
    ranked = sorted(instance_views.items(), key=lambda kv: len(kv[1]), reverse=True)
    ranked = ranked[: args.top_n]
    print(f"  After top-n={args.top_n} filter: {len(ranked)} instances")

    for inst_id, views in ranked:
        # optionally subsample
        if args.max_views > 0 and len(views) > args.max_views:
            step = max(1, len(views) // args.max_views)
            views = views[::step][: args.max_views]

        inst_dir = os.path.join(args.output_dir, f"instance_{inst_id:04d}")
        os.makedirs(inst_dir, exist_ok=True)

        saved = 0
        for frame, mask, fidx in views:
            crop = apply_mask_white_bg(frame, mask)
            if crop is None:
                continue
            out_path = os.path.join(inst_dir, f"frame_{fidx:06d}.jpg")
            cv2.imwrite(out_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            saved += 1

        print(f"  Saved instance {inst_id} ({saved} frames) -> {inst_dir}/")

    print(f"\nDone. Results in {args.output_dir}")
