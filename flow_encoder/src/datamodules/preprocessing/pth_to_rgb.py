"""This script converts pth flows to RGB flow frames."""
import argparse
from glob import glob
import os
import os.path as osp
from typing import Tuple

import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.file_utils import create_dir, load_pth
from utils.flow_utils import FlowUtils


def parse_arguments() -> Tuple[str, str]:
    """Parse input arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "flow_dir",
        type=str,
        help="Path to the directory containing subdir with pth flows",
    )
    parser.add_argument(
        "frame_dir",
        type=str,
        help="Path to the root saving directory RGB flow frames",
    )
    args = parser.parse_args()

    return args.flow_dir, args.frame_dir


if __name__ == "__main__":
    flow_rootdir, frame_rootdir = parse_arguments()
    # Iterate over the different directories containing pth flows
    for flow_dirname in tqdm(os.listdir(flow_rootdir)):
        flow_path_pattern = osp.join(flow_rootdir, flow_dirname, "*.pth")
        # Load pth flows
        pth_flows = [
            load_pth(flow_path)
            for flow_path in sorted(glob(flow_path_pattern))
        ]
        # Convert flows to RGB flow frames
        rgb_frames = [
            FlowUtils().flow_to_frame(flow.numpy()) for flow in pth_flows
        ]
        # Save flows as RGB flow frame
        frame_dir = osp.join(frame_rootdir, flow_dirname)
        create_dir(frame_dir)
        for k in range(len(rgb_frames)):
            flow_filename = osp.join(frame_dir, str(k).zfill(4) + ".png")
            plt.imsave(flow_filename, rgb_frames[k])
