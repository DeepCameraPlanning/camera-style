import argparse
import os
import os.path as osp

import numpy as np
from tqdm import tqdm

from features.motion_detector import FlowEstimator
from src.utils.tools import load_frames, load_pickle, write_clip


def parse_arguments():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "flow_dir",
        type=str,
        help="Path to the rootdir with precomputed forward and backward flows",
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

    flow_estimator = FlowEstimator(batch_size=1)
    for video_id in tqdm(os.listdir(flow_dir)):
        annotated_flow_path = osp.join(save_dir, video_id[:-2] + "mp4")

        flow_path = osp.join(flow_dir, video_id)
        flows = load_pickle(flow_path)

        annotated_flow = [flow_estimator.flow_to_frame(f) for f in flows]

        write_clip(annotated_flow, annotated_flow_path, fps=25)

# ffmpeg \
#     -i annotated_flows/raw_traveling.mp4 \
#     -i videos/high_traveling.mp4 \
#     -i annotated_flows/high_traveling.mp4 \
#     -i annotated_flows/low_traveling.mp4 \
#     -i annotated_flows/upscaled_traveling.mp4 \
#     -filter_complex "[0:v]scale=960:540[v0];[1:v]scale=960:540[v1];[2:v]scale=960:540[v2];[3:v]scale=960:540[v3];[4:v]scale=960:540[v4];[v0][v1][v2][v3][v4]vstack=inputs=5[v]" \
#     -map "[v]" output.mp4
