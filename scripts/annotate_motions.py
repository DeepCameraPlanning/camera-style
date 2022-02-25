import argparse
import os
import os.path as osp

import cv2

from features.motion_detector import FlowEstimator, MotionDetector
from src.utils.utils import write_clip, load_frames, load_pickle


encoding_to_motion = {
    0: "x_translation",
    1: "y_translation",
    2: "z_rotation",
    3: "z_translation",
    4: "null",
}
font = cv2.FONT_HERSHEY_DUPLEX
scale = 2
thickness = 5


def parse_arguments():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "root_dir",
        type=str,
        help="Path to the data root directory",
    )
    parser.add_argument(
        "--video-ids",
        "-v",
        type=str,
        default=None,
        help="Name of an optional list of video ids",
    )
    parser.add_argument(
        "--flow-rootdir",
        "-f",
        type=str,
        default=None,
        help="Path to the rootdir with precomputed forward and backward flows",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()
    root_dir = args.root_dir
    video_ids_filename = args.video_ids
    flow_rootdir = args.flow_rootdir

    if video_ids_filename is None:
        video_ids = os.listdir(osp.join(root_dir, "videos"))
    else:
        video_ids = load_pickle(video_ids_filename)

    motion_detector = MotionDetector(batch_size=128)
    if flow_rootdir is None:
        flow_estimator = FlowEstimator(batch_size=128)

    for video_id in video_ids:
        annotated_path = osp.join(root_dir, "annotated_motion", video_id)
        if osp.exists(annotated_path):
            continue

        video_path = osp.join(root_dir, "videos", video_id)
        frames = load_frames(video_path)

        if flow_rootdir is None:
            forward_flows = flow_estimator.estimate_flow(frames)
            backward_flows = flow_estimator.estimate_flow(frames[::-1])[::-1]
        else:
            forward_flow_path = osp.join(
                flow_rootdir, "forward_flows", video_id[:-3] + "pk"
            )
            forward_flows = load_pickle(forward_flow_path)
            backward_flow_path = osp.join(
                flow_rootdir, "backward_flows", video_id[:-3] + "pk"
            )
            backward_flows = load_pickle(backward_flow_path)

        detected_motions = motion_detector.detect_motion(
            forward_flows, backward_flows
        )

        annotated_frames = []
        for frame_index in range(len(detected_motions)):
            category_str = encoding_to_motion[detected_motions[frame_index]]
            annotated_frames.append(
                cv2.putText(
                    frames[frame_index],
                    category_str,
                    (20, 70),
                    font,
                    scale,
                    (0, 0, 255),
                    thickness,
                    cv2.LINE_AA,
                )
            )
        write_clip(annotated_frames, annotated_path, 25)
