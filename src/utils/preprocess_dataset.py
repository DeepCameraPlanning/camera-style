"""
This script preprcess original videos in `original_video_dir`, and save all
results in `preprocessed_dir`.

The preprocessind steps are:
    - Change the frame rate to 25fps, and save in `resampled_video_dir`
    - Extract audio tracks, and save in `audio_track_dir`
    - Extract all resampled video frames, and save in `frame_video_dir`
    - Extract shot boundaries, and save in `processed_shot_dir`
    - Extract and preprocess body and face tracks, and save in
        `output_bodytrack_path` and `output_facetrack_path`.
"""

import argparse
import os
import os.path as osp
import subprocess
from typing import List, Tuple

import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.cm_features import preprocess_bodytracks, preprocess_facetracks
from src.utils.utils import (
    create_dir,
    move_files,
    load_csv,
    load_pickle,
    save_csv,
    save_pickle,
)


def parse_arguments() -> Tuple[str, str]:
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root_dir",
        type=str,
        help="Dataset root directory",
    )
    parser.add_argument(
        "preprocessed_dir",
        type=str,
        help="Directory to store processing results",
    )
    parser.add_argument(
        "--video-paths",
        "-p",
        type=str,
        default=None,
        help="Directory to store processing results",
    )
    parser.add_argument(
        "--tracks",
        "-t",
        action="store_true",
        help="If provided, preprocess body and face tracks",
    )
    args = parser.parse_args()

    return (
        args.root_dir,
        args.preprocessed_dir,
        args.video_paths,
        args.tracks,
    )


def resample_video(input_video_path: str, output_video_path: str):
    """Change the frame rate to 25fps."""
    subprocess.call(
        [
            "ffmpeg",
            "-i",
            input_video_path,
            "-filter:v",
            "fps=25",
            output_video_path,
        ]
    )


def extract_audio(input_video_path: str, output_audio_path: str):
    """Extract audio track from the input video."""
    subprocess.call(
        [
            "ffmpeg",
            "-i",
            input_video_path,
            "-q:a",
            "0",
            "-map",
            "a",
            output_audio_path,
        ]
    )


def extract_frames(input_video_path: str, output_frame_dir: str):
    """Extract all input video frames."""
    create_dir(output_frame_dir)
    subprocess.call(
        [
            "ffmpeg",
            "-i",
            input_video_path,
            "-vf",
            "scale=min(iw*320/ih\,320):min(320\,ih*320/iw), "
            + "pad=320:320:(320-iw)/2:(320-ih)/2",
            f"{output_frame_dir}/%05d.png",
        ]
    )


def extract_shot(
    input_video_path: str,
    input_shot_path: str,
    output_shot_path: str,
):
    """
    Extract shot boundaries from Vision Movie dataset annotation file. Shot
    boundaries correspond to indices of shots' last frames.
    """
    clip = cv2.VideoCapture(input_video_path)
    fps = clip.get(5)

    raw_shot_df = pd.read_csv(input_shot_path, header=None, sep=" ")
    # Add +2 empirically (not understood why)
    raw_shot_boundaries = raw_shot_df[1].copy().to_numpy() + 2

    # Convert indices to 25fps indices and shift to have shots' last indices
    shot_boundaries = ((raw_shot_boundaries * 25 / fps).round()).astype(int)

    save_csv(shot_boundaries, output_shot_path)


def extract_bodytracks(
    input_video_path: str,
    input_bodytrack_path: str,
    output_bodytrack_path: str,
    processed_shot_path: str,
):
    """Extract pre-computed bodytracks."""
    clip = cv2.VideoCapture(input_video_path)
    fps = clip.get(5)
    if not osp.exists(input_bodytrack_path):
        print(input_bodytrack_path)
        preprocessed_bodytracks = []
    else:
        raw_bodytracks = load_pickle(input_bodytrack_path)
        shot_boundaries = load_csv(processed_shot_path)[0].to_list()
        preprocessed_bodytracks = preprocess_bodytracks(
            raw_bodytracks, shot_boundaries, threshold_iou=0.5, current_fps=fps
        )
    save_pickle(preprocessed_bodytracks, output_bodytrack_path)


def extract_facetracks(
    database_path: str,
    facedets_path: str,
    output_facetrack_path: str,
    processed_shot_path: str,
):
    """Extract pre-computed facetracks."""
    if not osp.exists(database_path):
        print(database_path)
        preprocessed_facetracks = []
    else:
        database = load_pickle(database_path)
        raw_facedets = load_pickle(facedets_path)
        shot_boundaries = load_csv(processed_shot_path)[0].to_list()
        preprocessed_facetracks = preprocess_facetracks(
            database, raw_facedets, shot_boundaries, current_fps=25
        )

    save_pickle(preprocessed_facetracks, output_facetrack_path)


def get_split(all_videos: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """Get the train, val and test lists of videos (0.6/0.2/0.2)."""
    train_videos, test_videos = train_test_split(
        all_videos, test_size=0.2, random_state=1
    )
    train_videos, val_videos = train_test_split(
        train_videos, test_size=0.25, random_state=1
    )
    return train_videos, val_videos, test_videos


def read_video_paths(video_paths: str) -> List[str]:
    """Read the list of video paths to process."""
    video_path_df = pd.read_csv(video_paths, header=None, sep=";")
    video_paths = video_path_df[0].tolist()
    return video_paths


if __name__ == "__main__":
    # Preprocessing directories
    (
        root_dir,
        preprocessed_dir,
        video_paths,
        tracks,
    ) = parse_arguments()

    original_video_dir = osp.join(root_dir, "videos")
    original_shot_dir = osp.join(root_dir, "shots", "raw")

    resampled_video_dir = osp.join(preprocessed_dir, "resampled_clips")
    audio_track_dir = osp.join(preprocessed_dir, "audio_tracks")
    frame_video_dir = osp.join(preprocessed_dir, "frame_clips")
    processed_shot_dir = osp.join(preprocessed_dir, "shot_boundaries")

    create_dir(preprocessed_dir)
    create_dir(resampled_video_dir)
    create_dir(audio_track_dir)
    create_dir(frame_video_dir)
    create_dir(processed_shot_dir)

    if tracks:
        original_bodytrack_dir = osp.join(root_dir, "bodytracks")
        original_facetrack_dir = osp.join(root_dir, "facetracks")
        processed_bodytrack_dir = osp.join(preprocessed_dir, "bodytracks")
        processed_facetrack_dir = osp.join(preprocessed_dir, "facetracks")
        create_dir(processed_bodytrack_dir)
        create_dir(processed_facetrack_dir)

    # Train, val and test split directories
    if video_paths:
        all_videos = read_video_paths(video_paths)
    else:
        all_videos = os.listdir(original_video_dir)
    train_videos, val_videos, test_videos = get_split(all_videos)

    frame_train_dir = osp.join(frame_video_dir, "train")
    frame_val_dir = osp.join(frame_video_dir, "val")
    frame_test_dir = osp.join(frame_video_dir, "test")

    audio_train_dir = osp.join(audio_track_dir, "train")
    audio_val_dir = osp.join(audio_track_dir, "val")
    audio_test_dir = osp.join(audio_track_dir, "test")

    create_dir(frame_train_dir)
    create_dir(frame_val_dir)
    create_dir(frame_test_dir)
    create_dir(audio_train_dir)
    create_dir(audio_val_dir)
    create_dir(audio_test_dir)
    for video_filename in sorted(all_videos):
        video_filename = video_filename[:-4]
        video_year, video_id = video_filename.split(osp.sep)
        original_video_path = osp.join(original_video_dir, video_filename)

        # Resample the video
        resample_filename = video_id + ".mkv"
        resampled_video_path = osp.join(resampled_video_dir, resample_filename)
        resample_video(original_video_path, resampled_video_path)

        # Extract the audio track
        audio_track_path = osp.join(audio_track_dir, video_id + ".mp3")
        extract_audio(resampled_video_path, audio_track_path)

        # Extract resampled video frames
        frame_dir = osp.join(frame_video_dir, video_id)
        extract_frames(resampled_video_path, frame_dir)

        # Extract shot boundaries
        processed_shot_path = osp.join(processed_shot_dir, video_id + ".csv")
        original_shot_path = osp.join(
            original_shot_dir,
            video_year,
            video_id,
            "shot_txt",
            video_id + ".txt",
        )
        extract_shot(
            original_video_path,
            original_shot_path,
            processed_shot_path,
        )

        if tracks:
            # Extract bodytracks
            original_bodytrack_path = osp.join(
                original_bodytrack_dir, video_year, video_id + ".pkl"
            )
            processed_bodytrack_path = osp.join(
                processed_bodytrack_dir, video_id + ".pkl"
            )
            extract_bodytracks(
                original_video_path,
                original_bodytrack_path,
                processed_bodytrack_path,
                processed_shot_path,
            )
            # Extract facetracks
            facetrack_path = osp.join(
                original_facetrack_dir, video_year, video_id + ".mkv"
            )
            database_path = facetrack_path + "database.pk"
            facedets_path = facetrack_path + "face_dets.pk"
            processed_bodytrack_path = osp.join(
                processed_facetrack_dir, video_id + ".pkl"
            )
            extract_facetracks(
                database_path,
                facedets_path,
                processed_bodytrack_path,
                processed_shot_path,
            )

        # Move frame and audio files inside train, val and test directories
        if video_filename in train_videos:
            move_files(frame_dir, frame_train_dir)
            move_files(audio_track_path, audio_train_dir)
        elif video_filename in val_videos:
            move_files(frame_dir, frame_val_dir)
            move_files(audio_track_path, audio_val_dir)
        else:
            move_files(frame_dir, frame_test_dir)
            move_files(audio_track_path, audio_test_dir)
