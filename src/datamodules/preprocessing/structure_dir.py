"""
This script copies flow and RGB frames from a raw Unity output (`raw_all`) to
specific flows and frames directories (`flow_frames` and `raw_frames`). It
also renames files to keep only the frame index.
"""

import argparse
import os
import os.path as osp
from typing import Tuple
import subprocess

from tqdm import tqdm

from src.utils.file_utils import create_dir


def parse_arguments() -> Tuple[str, str]:
    """Parse input arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "root_dir",
        type=str,
        help="Path to the directory containing `raw_all`, `flow_frames` and "
        + "`raw_frames` directories.",
    )

    args = parser.parse_args()

    return args.root_dir


def copy_rename_files(all_dir: str, flow_dir: str, frame_dir: str):
    """Copiy and rename flow and RGB frames."""
    for filename in tqdm(sorted(os.listdir(all_dir))):
        if "_flow.png" in filename:
            flow_filename = filename.split("_")[1] + ".png"
            raw_all_path = osp.join(all_dir, filename)
            flow_frame_path = osp.join(flow_dir, flow_filename)
            subprocess.call(["cp", raw_all_path, flow_frame_path])

        elif "_img.png" in filename:
            frame_filename = filename.split("_")[1] + ".png"
            raw_all_path = osp.join(all_dir, filename)
            raw_frame_path = osp.join(frame_dir, frame_filename)
            subprocess.call(["cp", raw_all_path, raw_frame_path])


if __name__ == "__main__":
    root_dir = parse_arguments()
    root_all_dir = osp.join(root_dir, "raw_all")
    for clip_dirname in os.listdir(root_all_dir):
        raw_all_dir = osp.join(root_all_dir, clip_dirname)
        flow_frames_dir = osp.join(root_dir, "flow_frames", clip_dirname)
        raw_frames_dir = osp.join(root_dir, "raw_frames", clip_dirname)

        create_dir(flow_frames_dir)
        create_dir(raw_frames_dir)

        copy_rename_files(raw_all_dir, flow_frames_dir, raw_frames_dir)
