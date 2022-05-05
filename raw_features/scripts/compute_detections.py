import argparse
import os
import os.path as osp
from typing import Any, List

from tqdm import tqdm

from raw_features.core.people_detector import PeopleDetector
from utils.file_utils import create_dir, save_pickle, load_frames, load_pickle


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


def split_chunks(array: List[Any], chunk_size: int):
    """Yield successive n-sized chunks from `array`."""
    for i in range(0, len(array), chunk_size):
        yield array[i : i + chunk_size]


if __name__ == "__main__":
    args = parse_arguments()
    video_dir = args.video_dir
    video_ids_filename = args.video_ids

    save_dir = args.save_dir
    create_dir(save_dir)
    raw_detections_dir = osp.join(save_dir, "raw_detections")
    create_dir(raw_detections_dir)
    tracked_detections_dir = osp.join(save_dir, "tracked_detections")
    create_dir(tracked_detections_dir)

    if video_ids_filename is None:
        video_ids = os.listdir(video_dir)
    else:
        video_ids = load_pickle(video_ids_filename)

    people_detector = PeopleDetector(batch_size=16, verbose=True)
    for video_id in tqdm(video_ids):
        video_path = osp.join(video_dir, video_id)
        save_name = video_id[:-3] + "pk"
        frames = load_frames(video_path)
        height, width, _ = frames[0].shape

        raw_detections_path = osp.join(raw_detections_dir, save_name)
        if osp.exists(raw_detections_path):
            continue

        bboxes, masks, scores = [], [], []
        for frame_chunk in split_chunks(frames, 300):
            raw_detections = people_detector.detect_people(frame_chunk)
            bboxes.extend(raw_detections[0])
            masks.extend(raw_detections[1])
            scores.extend(raw_detections[2])
        raw_detections = (bboxes, masks, scores)

        save_pickle([(height, width), raw_detections], raw_detections_path)

        tracked_detections_path = osp.join(tracked_detections_dir, save_name)
        bboxes, masks, scores = raw_detections
        bbox_tracks = people_detector.get_bboxtracks(bboxes, scores)
        save_pickle(((height, width), bbox_tracks), tracked_detections_path)
