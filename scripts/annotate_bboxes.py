import argparse
import os
import os.path as osp
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm

from src.utils.utils import load_frames, load_pickle, write_clip


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


def draw_bbox(
    frame: np.array,
    bbox: np.array,
    text: str = None,
    color: Tuple[int, int, int] = (26, 28, 228),
) -> np.array:
    """Draw a bounding box.
    :param frame: frame to annotate.
    :param bbox: frame index and bounding-box coordinates.
    :param text: text to write inside the bounding-box at the top-left,
        write only if provided.
    :param color: color of the bounding-box (default red).
    :return: annotated frame.
    """
    # Draw bbox
    frame = cv2.rectangle(
        frame,
        (bbox[1], bbox[2]),
        (bbox[3], bbox[4]),
        color,
        thickness=5,
    )
    if text:
        # Write character's id and name inside the bbox
        frame = cv2.putText(
            frame,
            text,
            (bbox[1] + 5, bbox[2] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=color,
            thickness=2,
        )

    return frame


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

    for video_id in tqdm(video_ids):
        video_path = osp.join(root_dir, "videos", video_id)
        frames = load_frames(video_path)

        tracked_detections_path = osp.join(
            root_dir, "tracked_detections", video_id[:-3] + "pk"
        )
        bbox_tracks = load_pickle(tracked_detections_path)

        annotated_frames = []
        all_prev_x = [None for _ in range(len(bbox_tracks))]
        all_prev_y = [None for _ in range(len(bbox_tracks))]
        for frame_index, frame in enumerate(frames):
            for track_index, track in enumerate(bbox_tracks):
                (bbox_index,) = np.where(track[:, 0] == frame_index)
                bbox = track[bbox_index].reshape(-1).astype(int)
                if bbox.size > 0:
                    color = COLORS[track_index % len(COLORS)].tolist()

                    x = int(bbox[1] + ((bbox[3] - bbox[1]) / 2))
                    y = int(bbox[2] + ((bbox[4] - bbox[2]) / 2))
                    prev_x = all_prev_x[track_index]
                    prev_y = all_prev_y[track_index]
                    if prev_x is None:
                        prev_x = x
                        prev_y = y

                    frame = cv2.circle(frame, (x, y), 3, color, -1)
                    frame = cv2.circle(frame, (prev_x, prev_y), 3, color, -1)
                    frame = cv2.line(frame, (x, y), (prev_x, prev_y), color)
                    frame = draw_bbox(frame, bbox, color=color)

                    all_prev_x[track_index] = x
                    all_prev_y[track_index] = y

            annotated_frames.append(frame)

        annotated_bbox_path = osp.join(root_dir, "annotated_bboxes", video_id)
        write_clip(annotated_frames, annotated_bbox_path)
