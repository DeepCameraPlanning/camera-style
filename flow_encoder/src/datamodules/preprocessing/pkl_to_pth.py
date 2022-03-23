"""
This script splits entire video flow fields to frame-wise flow fields.
"""
import argparse
import os
import os.path as osp
from typing import Tuple

from tqdm import tqdm
import torch

from utils.file_utils import create_dir, load_pickle, save_pth


def parse_arguments() -> Tuple[str, str]:
    """Parse input arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "pk_dir",
        type=str,
        help="Path to the directory containing the pickle files",
    )
    parser.add_argument(
        "pth_dir",
        type=str,
        help="Path to the root flow saving directory",
    )
    args = parser.parse_args()

    return args.pk_dir, args.pth_dir


if __name__ == "__main__":
    pk_dir, pth_dir = parse_arguments()

    for pk_filename in tqdm(os.listdir(pk_dir)):
        # Load whole video flows
        pk_path = osp.join(pk_dir, pk_filename)
        flows = load_pickle(pk_path)

        # Save flows as pytorch tensors
        flow_dir = osp.join(pth_dir, pk_filename[:-3])
        create_dir(flow_dir)
        for k in range(len(flows)):
            flow_filename = osp.join(flow_dir, str(k).zfill(4) + ".pth")
            flow_tensor = torch.from_numpy(flows[k])
            save_pth(flow_tensor, flow_filename)
