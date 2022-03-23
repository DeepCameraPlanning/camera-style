import argparse
import os
import os.path as osp
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from raw_features.core.pose_detector import PoseEstimator
from raw_features.core.people_detector import PeopleDetector
from utils.file_utils import load_frames, load_pickle, write_clip
from lib.dope.visu import visualize_bodyhandface2d


def draw_pose(
    frame: np.array,
    poses: List[np.array],
    color: Tuple[int, int, int] = None,
) -> np.array:
    """
    Draw each detected 2D poses on a given frame.
    Code adapted from: https://github.com/naver/dope.
    """
    if color:
        # Linked joints indices
        bones = [
            [0, 2],  # Right ankle - knee
            [1, 3],  # Left ankle - knee
            [2, 4],  # Right knee - hip
            [3, 5],  # Left knee - hip
            [4, 5],  # Right - left hip
            [6, 8],  # Right wrist - elbow
            [7, 9],  # Left wrist - elbow
            [8, 10],  # Right elbow - shoulder
            [9, 11],  # Left elbow - shoulder
            [10, 11],  # Right - left shoulder
            [12, 13],  # Head - neck
            [13, 14],  # Neck - pelvis
        ]

        for poselet in poses:
            neck_joint = poselet[[10, 11]].mean(axis=0).astype(int)
            pelvis_joint = poselet[[4, 5]].mean(axis=0).astype(int)
            poselet = poselet.astype(int).tolist()
            poselet.extend([neck_joint, pelvis_joint])
            # Draw joints
            for x_joint, y_joint in poselet:
                cv2.circle(frame, (x_joint, y_joint), 5, color, -1)
            # Draw bones
            for joint_index_1, joint_index_2 in bones:
                x_joint_1, y_joint_1 = poselet[joint_index_1]
                x_joint_2, y_joint_2 = poselet[joint_index_2]
                cv2.line(
                    frame,
                    (x_joint_1, y_joint_1),
                    (x_joint_2, y_joint_2),
                    color,
                    3,
                )

        return frame

    # Get detected 2D poses of each detected part
    detected_poses2d = {"body": poses}
    pose_scores = {"body": [1 for _ in poses]}

    # Draw 2D poses on the given frame
    annotated_frame = visualize_bodyhandface2d(
        frame, detected_poses2d, dict_scores=pose_scores, max_padding=0
    )

    return annotated_frame


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
    args = parser.parse_args()

    return args


COLORS = np.array(
    [
        [28, 26, 228],
        [184, 126, 55],
        [74, 175, 77],
        [163, 78, 152],
        [0, 127, 255],
        [51, 255, 255],
        [40, 86, 166],
    ]
).astype(np.uint8)

if __name__ == "__main__":
    args = parse_arguments()
    root_dir = args.root_dir
    video_ids_filename = args.video_ids

    if video_ids_filename is None:
        video_ids = os.listdir(osp.join(root_dir, "videos"))
    else:
        video_ids = load_pickle(video_ids_filename)

    pose_estimator = PoseEstimator(batch_size=32)
    people_detector = PeopleDetector(batch_size=32)
    for video_id in tqdm(video_ids):
        video_path = osp.join(root_dir, "videos", video_id)
        frames = load_frames(video_path)
        frame_dims = frames[0].shape[:2]
        estimated_poses = pose_estimator.estimate_pose(frames)
        bboxes, _, scores = people_detector.detect_people(frames)
        bbox_tracks = people_detector.get_bboxtracks(bboxes, scores)
        tracked_poses = pose_estimator.track_poselets(
            estimated_poses, bbox_tracks, frame_dims
        )

        annotated_frames = []
        for frame_index, frame in enumerate(frames):
            frame = frames[frame_index]
            for track_index, track in enumerate(tracked_poses):
                (pose_index,) = np.where(track[:, 0] == frame_index)
                pose = track[pose_index, 1]
                if pose.size > 0:
                    draw_pose(frame, pose, COLORS[track_index])
            annotated_frames.append(frame)

        annotated_poses_path = osp.join(root_dir, "annotated_poses", video_id)
        write_clip(annotated_frames, annotated_poses_path)
