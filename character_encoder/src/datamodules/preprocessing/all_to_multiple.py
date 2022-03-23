"""This script splits extracted feature file to clip-wise feature files."""
import argparse
from collections import defaultdict
import os.path as osp
from typing import Tuple

from tqdm import tqdm

from utils.file_utils import create_dir, load_pickle, save_pickle


def parse_arguments() -> Tuple[str, str]:
    """Parse input arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "pk_path",
        type=str,
        help="Path to the pickle file containing all clip features",
    )
    parser.add_argument(
        "save_dir",
        type=str,
        help="Path to the saving directory",
    )
    args = parser.parse_args()

    return args.pk_path, args.save_dir


if __name__ == "__main__":
    pk_path, save_dir = parse_arguments()
    create_dir(save_dir)

    # Load whole clip features
    flow_features = load_pickle(pk_path)

    # Gather features per clip
    clip_to_features = defaultdict(dict)
    for chunk_name, features in flow_features.items():
        clip_name, frame_indices = chunk_name.split("/")
        clip_to_features[clip_name][frame_indices] = features
    clip_to_features = dict(clip_to_features)

    # Save each clip feature into a single file
    for clip_name, chunk_features in tqdm(clip_to_features.items()):
        clip_save_dir = osp.join(save_dir, clip_name)
        create_dir(clip_save_dir)
        for frame_indices, feature in chunk_features.items():
            save_path = osp.join(clip_save_dir, frame_indices + ".pk")
            save_pickle(feature, save_path)
