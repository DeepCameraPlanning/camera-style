import argparse
import os
import os.path as osp

from tqdm import tqdm

from features.people_detector import PeopleDetector
from src.utils.utils import save_pickle, load_frames, load_pickle


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


if __name__ == "__main__":
    args = parse_arguments()
    root_dir = args.root_dir
    video_ids_filename = args.video_ids

    if video_ids_filename is None:
        video_ids = os.listdir(osp.join(root_dir, "videos"))
    else:
        video_ids = load_pickle(video_ids_filename)

    people_detector = PeopleDetector(batch_size=32)
    for video_id in tqdm(video_ids):
        video_path = osp.join(root_dir, "videos", video_id)
        frames = load_frames(video_path)

        raw_detections_path = osp.join(
            "./data", "raw_detections", video_id[:-3] + "pk"
        )
        if not osp.exists(raw_detections_path):
            raw_detections = people_detector.detect_people(frames)
            save_pickle(raw_detections, raw_detections_path)

        tracked_detections_path = osp.join(
            "./data", "tracked_detections", video_id[:-3] + "pk"
        )
        if not osp.exists(tracked_detections_path):
            bboxes, masks, scores = raw_detections
            bbox_tracks = people_detector.get_bboxtracks(bboxes, scores)
            save_pickle(bbox_tracks, tracked_detections_path)
