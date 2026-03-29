#!/usr/bin/env python3
"""
End-to-end demo: MP4 -> segmentation inference -> output MP4

Usage:
    python visualization/demo_mp4.py \
        --input video.mp4 \
        --output result.mp4 \
        --config-file configs/ytvis21/video_maskformer2_R50_bs32_8ep.yaml \
        --opts MODEL.WEIGHTS /path/to/weights.pth
"""

import argparse
import glob
import multiprocessing as mp
import os
import shutil
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


def get_video_fps(video_path):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        capture_output=True,
        text=True,
    )
    rate = result.stdout.strip()
    if "/" in rate:
        num, den = rate.split("/")
        return float(num) / float(den)
    return float(rate) if rate else 30.0


def extract_frames(video_path, frames_dir):
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, os.path.join(frames_dir, "frame_%06d.jpg")],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def assemble_mp4(frames_dir, output_path, fps):
    frame_pattern = os.path.join(frames_dir, "frame_%06d.jpg")
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
    parser = argparse.ArgumentParser(description="End-to-end MP4 video segmentation demo")
    parser.add_argument("--input", required=True, help="Path to input MP4 file")
    parser.add_argument("--output", required=True, help="Path for output MP4 file")
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

    assert os.path.isfile(args.input), f"Input file not found: {args.input}"

    cfg = setup_cfg(args)
    demo = VisualizationDemo_windows(cfg)

    fps = get_video_fps(args.input)
    logger.info(f"Detected video FPS: {fps:.2f}")

    with tempfile.TemporaryDirectory() as tmpdir:
        frames_dir = os.path.join(tmpdir, "frames")
        vis_dir = os.path.join(tmpdir, "vis")
        os.makedirs(frames_dir)
        os.makedirs(vis_dir)

        # Step 1: extract frames
        logger.info("Extracting frames from MP4...")
        extract_frames(args.input, frames_dir)

        frames_path = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
        logger.info(f"Extracted {len(frames_path)} frames")

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
        # rename vis frames to sequential pattern expected by ffmpeg
        vis_frames = sorted(glob.glob(os.path.join(vis_dir, "*.jpg")))
        seq_dir = os.path.join(tmpdir, "seq")
        os.makedirs(seq_dir)
        for idx, f in enumerate(vis_frames):
            os.symlink(f, os.path.join(seq_dir, f"frame_{idx+1:06d}.jpg"))

        assemble_mp4(seq_dir, args.output, fps)

    logger.info(f"Done. Output saved to: {args.output}")
