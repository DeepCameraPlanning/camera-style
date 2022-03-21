import argparse
import os
import os.path as osp
from typing import Tuple

from tqdm import tqdm

from src.datamodules.augmentations.reverse import reverse_clip, reverse_flow
from src.datamodules.augmentations.speedup import speedup_clip, speedup_flow


def parse_arguments() -> Tuple[str]:
    """Parse input arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "raw_dir",
        type=str,
        help="Path to the directory containing the raw frame directories",
    )
    parser.add_argument(
        "save_dir",
        type=str,
        help="Path to the directory to the augmented raw frames",
    )
    parser.add_argument(
        "flow_dir",
        type=str,
        help="Path to the directory containing the pth unity flow directories",
    )
    parser.add_argument(
        "--reverse",
        "-r",
        action="store_true",
        help="Wether to reverse clips or not",
    )
    parser.add_argument(
        "--speedup",
        "-s",
        action="store_true",
        help="Wether to speedup clips or not",
    )

    args = parser.parse_args()

    return (
        args.raw_dir,
        args.save_dir,
        args.flow_dir,
        args.reverse,
        args.speedup,
    )


if __name__ == "__main__":
    raw_dir, save_dir, flow_dir, reverse, speedup = parse_arguments()

    # Reverse
    if reverse:
        for clip_dirname in tqdm(os.listdir(raw_dir)):
            input_raw_dir = osp.join(raw_dir, clip_dirname)
            input_flow_dir = osp.join(flow_dir, clip_dirname)
            reversed_raw_dir = osp.join(save_dir, "reversed_" + clip_dirname)
            reverse_clip(input_raw_dir, reversed_raw_dir)
            reversed_flow_dir = osp.join(flow_dir, "reversed_" + clip_dirname)
            reverse_flow(input_flow_dir, reversed_flow_dir)

    # Speedup
    if speedup:
        for clip_dirname in tqdm(os.listdir(raw_dir)):
            speedup_raw_dir = osp.join(save_dir, "speedup_" + clip_dirname)
            speedup_clip(input_raw_dir, speedup_raw_dir)
            speedup_flow_dir = osp.join(flow_dir, "speedup_" + clip_dirname)
            speedup_flow(input_flow_dir, speedup_flow_dir)
