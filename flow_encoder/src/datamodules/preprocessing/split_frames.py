"""
This script splits all videos contained in an input directory into `.png`
frames, and save frame directories into a saving directory.
"""
import argparse
import os
import os.path as osp
import subprocess
from typing import Tuple

from tqdm import tqdm

from src.utils.file_utils import create_dir


def split_clip(clip_path: str, frame_dir: str):
    """ "
    Split a clip at `clip_path` into `.png` frames, and save them in
    `frame_dir`.
    """
    saving_path = osp.join(frame_dir, "%04d.png")
    subprocess.call(["ffmpeg", "-i", clip_path, saving_path, "-hide_banner"])


def parse_arguments() -> Tuple[str, str]:
    """Parse input arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "clip_dir",
        type=str,
        help="Path to the directory containing all the clips to split",
    )
    parser.add_argument(
        "saving_dir",
        type=str,
        help="Path to the saving directory",
    )
    args = parser.parse_args()

    return args.clip_dir, args.saving_dir


if __name__ == "__main__":
    clip_dir, saving_dir = parse_arguments()
    create_dir(saving_dir)

    for clip_filename in tqdm(os.listdir(clip_dir)):
        clip_path = osp.join(clip_dir, clip_filename)
        frame_dir = osp.join(saving_dir, clip_filename.split(".")[0])
        create_dir(frame_dir)
        split_clip(clip_path, frame_dir)
