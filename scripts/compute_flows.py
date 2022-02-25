import argparse
import os
import os.path as osp

from tqdm import tqdm

from features.motion_detector import FlowEstimator
from src.utils.tools import save_pickle, load_frames, load_pickle


def parse_arguments():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "video_dir",
        type=str,
        help="Path to the data root directory",
    )
    parser.add_argument(
        "save_dir",
        type=str,
        help="Path to the saving directory",
    )
    parser.add_argument(
        "--video-ids",
        "-v",
        type=str,
        default=None,
        help="Name of an optional list of video ids",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()
    video_dir = args.video_dir
    save_dir = args.save_dir
    video_ids_filename = args.video_ids

    if video_ids_filename is None:
        video_ids = os.listdir(video_dir)
    else:
        video_ids = load_pickle(video_ids_filename)

    flow_estimator = FlowEstimator(batch_size=16)
    for video_id in tqdm(video_ids):
        video_path = osp.join(video_dir, video_id)
        frames = load_frames(video_path)

        forward_flow_path = osp.join(
            save_dir, "forward_flows", video_id[:-3] + "pk"
        )
        # if not osp.exists(forward_flow_path):
        forward_flows = flow_estimator.estimate_flow((frames))
        save_pickle(forward_flows, forward_flow_path)

        backward_flow_path = osp.join(
            save_dir, "backward_flows", video_id[:-3] + "pk"
        )
        # if not osp.exists(backward_flow_path):
        backward_flows = flow_estimator.estimate_flow(frames[::-1])[::-1]
        save_pickle(backward_flows, backward_flow_path)
