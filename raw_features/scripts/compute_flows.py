import argparse
import os
import os.path as osp

import cv2
from tqdm import tqdm

from utils.file_utils import (
    create_dir,
    save_pickle,
    load_pickle,
    load_frames_fromdir,
)
from raw_features.core.motion_detector import FlowEstimator


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
        "--model-name",
        "-m",
        type=str,
        default="raft",
        help="Name of optical flow model",
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


# 2x Nested clips /data/dir/video_id/shot_index
if __name__ == "__main__":
    args = parse_arguments()
    frame_dir = args.video_dir
    save_dir = args.save_dir
    model_name = args.model_name
    video_ids_filename = args.video_ids

    create_dir(save_dir)
    if video_ids_filename is None:
        video_ids = os.listdir(frame_dir)
    else:
        video_ids = load_pickle(video_ids_filename)
    orig_dir = "/media/hdd_robin/MovieShot/flows"
    flow_estimator = FlowEstimator(model_name=model_name, batch_size=1)

    for video_id in tqdm(sorted(video_ids)[:7200][::-1]):
        video_dir = osp.join(frame_dir, video_id)
        flow_dir = osp.join(save_dir, video_id)
        orig_flow_dir = osp.join(orig_dir, video_id)
        create_dir(flow_dir)
        for shot_index in sorted(os.listdir(video_dir)):
            video_path = osp.join(video_dir, shot_index)
            flow_path = osp.join(flow_dir, shot_index + ".pk")
            orig_flow_path = osp.join(orig_flow_dir, shot_index + ".pk")
            if osp.exists(orig_flow_path) or osp.exists(flow_path):
                continue
            # Load frames and convert them from BRG to RGB
            try:
                frames = [
                    cv2.cvtColor(cv2.resize(f, (224, 224)), cv2.COLOR_BGR2RGB)
                    for f in load_frames_fromdir(video_path)
                ]
                flows = flow_estimator.estimate_flow(frames)
                save_pickle(flows, flow_path)
            except:
                import ipdb

                ipdb.set_trace()


# 1x Nested clips /data/dir/video_id
# if __name__ == "__main__":
#     args = parse_arguments()
#     video_dir = args.video_dir
#     save_dir = args.save_dir
#     model_name = args.model_name
#     video_ids_filename = args.video_ids

#     create_dir(save_dir)
#     if video_ids_filename is None:
#         video_ids = os.listdir(video_dir)
#     else:
#         video_ids = load_pickle(video_ids_filename)

#     flow_estimator = FlowEstimator(model_name=model_name, batch_size=1)
#     for video_id in tqdm(video_ids):
#         video_path = osp.join(video_dir, video_id)
#         flow_path = osp.join(save_dir, video_id + ".pk")
#         # if osp.exists(flow_path):
#         #     continue
#         # Load frames and convert them from BRG to RGB
#         frames = [
#             cv2.cvtColor(cv2.resize(f, (224, 224)), cv2.COLOR_BGR2RGB)
#             for f in load_frames_fromdir(video_path)
#         ]
#         flows = flow_estimator.estimate_flow(frames)
#         save_pickle(flows, flow_path)
