import argparse
import os.path as osp
import subprocess

import cv2
import numpy as np
import pandas as pd

from src.utils.tools import load_pickle


def extract_shots(
    input_dir: str,
    output_dir: str,
    clip_year_ids: str,
    duration_threshold: int,
):
    """
    Extract shots greater than `duration_threshold` from the selected videos.
    """
    for year_id in clip_year_ids:
        clip_year, clip_id = year_id.split("/")
        shot_path = osp.join(
            input_dir,
            "shots",
            "raw",
            clip_year,
            clip_id,
            "shot_txt",
            clip_id + ".txt",
        )
        shots = pd.read_csv(shot_path, header=None, sep=" ").to_numpy()
        if shots.shape[0] > 1:
            shot_durations = shots[:, 1] - shots[:, 0]
        else:
            shot_durations = np.array([shots[0, 1]])
        longest_shots = shots[:, :2][shot_durations > duration_threshold]
        if longest_shots.shape[0] == 0:
            continue

        input_path = osp.join(input_dir, "videos", clip_year, clip_id + ".mkv")
        if osp.exists(input_path):
            continue
        for shot_index, (start_index, end_index) in enumerate(longest_shots):
            output_name = str(shot_index) + "_" + clip_id + ".mkv"
            output_path = osp.join(output_dir, "videos", output_name)
            extract_subclip(input_path, output_path, start_index, end_index)


def extract_subclip(
    input_path: str, output_path: str, start_index: int, end_index: int
):
    """Extract a subclip from a video."""
    input_clip = cv2.VideoCapture(input_path)
    fps = input_clip.get(5)
    frame_duration = end_index - start_index
    start_time = str(start_index / fps)
    duration_time = str(frame_duration / fps)

    subprocess.call(
        [
            "ffmpeg",
            "-ss",
            start_time,
            "-i",
            input_path,
            "-t",
            duration_time,
            "-async",
            "1",
            output_path,
        ]
    )


def parse_arguments():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to the clip input directory",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to the clip output directory",
    )
    parser.add_argument(
        "yearid_path",
        type=str,
        help="Path to year-id clip list",
    )
    parser.add_argument(
        "duration_threshold",
        type=int,
        help="Shot duration threshold",
        default=240,
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()
    input_dir = args.input_dir
    output_dir = args.output_dir
    clip_year_ids = load_pickle(args.yearid_path)
    duration_threshold = args.duration_threshold
    extract_shots(input_dir, output_dir, clip_year_ids, duration_threshold)
