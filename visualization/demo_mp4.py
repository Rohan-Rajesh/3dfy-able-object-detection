#!/usr/bin/env python3
"""
End-to-end demo: frames folder -> segmentation inference -> output MP4

Usage:
    python visualization/demo_mp4.py \
        --input /path/to/frames_dir \
        --output result.mp4 \
        --config-file configs/ytvis21/video_maskformer2_R50_bs32_8ep.yaml \
        --opts MODEL.WEIGHTS /path/to/weights.pth
"""

import argparse
import glob
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time

import torch
import tqdm

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

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


def assemble_mp4(frames_dir, output_path, fps, ext=".jpg"):
    frame_pattern = os.path.join(frames_dir, f"frame_%06d{ext}")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def get_parser():
    parser = argparse.ArgumentParser(description="End-to-end frames-folder segmentation demo")
    parser.add_argument("--input", required=True, help="Path to folder of input frames (jpg/png)")
    parser.add_argument("--output", required=True, help="Path for output MP4 file")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS for output MP4 (default: 30)")
    parser.add_argument(
        "--config-file",
        default="configs/ytvis21/video_maskformer2_R50_bs32_8ep.yaml",
        metavar="FILE",
        help="Path to model config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--windows-size",
        type=int,
        default=20,
        help="Window size for semi-offline inference mode (-1 = full video)",
    )
    parser.add_argument(
        "--opts",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra config options, e.g. MODEL.WEIGHTS /path/to/model.pth",
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    assert os.path.isdir(args.input), f"Input folder not found: {args.input}"

    cfg = setup_cfg(args)
    demo = VisualizationDemo_windows(cfg)

    fps = args.fps
    logger.info(f"Output FPS: {fps:.2f}")

    with tempfile.TemporaryDirectory() as tmpdir:
        vis_dir = os.path.join(tmpdir, "vis")
        os.makedirs(vis_dir)

        frames_path = sorted(
            glob.glob(os.path.join(args.input, "*.jpg")) +
            glob.glob(os.path.join(args.input, "*.png"))
        )
        logger.info(f"Found {len(frames_path)} frames")

        windows_size = args.windows_size
        if windows_size == -1:
            windows_size = len(frames_path)

        # Step 2: run inference
        logger.info("Running segmentation inference...")
        start_time = time.time()
        vid_frames, _frames_path, instances = [], [], set()

        for i, path in enumerate(tqdm.tqdm(frames_path)):
            img = read_image(path, format="BGR")
            _frames_path.append(path)
            vid_frames.append(img)

            if len(vid_frames) == windows_size or i == len(frames_path) - 1:
                with torch.amp.autocast(device_type="cuda"):
                    keep = i >= windows_size
                    predictions, visualized_output = demo.run_on_video(vid_frames, keep=keep)

                for p, vis in zip(_frames_path, visualized_output):
                    out_name = os.path.join(vis_dir, os.path.basename(p))
                    vis.save(out_name)

                if "pred_ids" in predictions:
                    for id_ in predictions["pred_ids"]:
                        instances.add(id_)

                del visualized_output, vid_frames, _frames_path, predictions
                vid_frames, _frames_path = [], []

        elapsed = time.time() - start_time
        logger.info(
            f"Detected {len(instances)} instances in {elapsed:.2f}s "
            f"({len(frames_path)/elapsed:.1f} fps)"
        )

        # Step 3: reassemble MP4
        logger.info(f"Assembling output MP4: {args.output}")
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        # rename vis frames to sequential pattern expected by ffmpeg
        vis_frames = sorted(
            glob.glob(os.path.join(vis_dir, "*.jpg")) +
            glob.glob(os.path.join(vis_dir, "*.png"))
        )
        seq_dir = os.path.join(tmpdir, "seq")
        os.makedirs(seq_dir)
        ext = os.path.splitext(vis_frames[0])[1] if vis_frames else ".jpg"
        for idx, f in enumerate(vis_frames):
            os.symlink(f, os.path.join(seq_dir, f"frame_{idx+1:06d}{ext}"))

        assemble_mp4(seq_dir, args.output, fps, ext)

    logger.info(f"Done. Output saved to: {args.output}")
