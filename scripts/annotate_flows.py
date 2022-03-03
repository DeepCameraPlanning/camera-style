import argparse
import os
import os.path as osp

import torch

from features.motion_detector import FlowEstimator
from src.utils.utils import create_dir, write_clip


def parse_arguments():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "flow_dir",
        type=str,
        help="Path to the rootdir with `.pth` flow frames",
    )
    parser.add_argument(
        "save_dir",
        type=str,
        help="Path to the saving directory",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()
    flow_dir = args.flow_dir
    save_dir = args.save_dir

    create_dir(save_dir)

    flow_estimator = FlowEstimator(batch_size=1)
    raw_flows = [
        torch.load(osp.join(flow_dir, flow_filename))
        for flow_filename in sorted(os.listdir(flow_dir))
    ]
    annotated_flow = [
        flow_estimator.flow_to_frame(f.numpy()) for f in raw_flows
    ]

    flow_filename = osp.basename(osp.normpath(flow_dir)) + ".mp4"
    annotated_flow_path = osp.join(save_dir, flow_filename)
    write_clip(annotated_flow, annotated_flow_path, fps=25)
