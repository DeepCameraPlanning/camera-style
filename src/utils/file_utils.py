import os
import os.path as osp
import pickle
import subprocess
from typing import Any, List

import cv2
import numpy as np
import pandas as pd
import torch


def create_dir(dir_name: str):
    """Create a directory if it does not exist yet."""
    if not osp.exists(dir_name):
        os.makedirs(dir_name)


def move_files(source_path: str, destpath: str):
    """Move files from `source_path` to `dest_path`."""
    subprocess.call(["mv", source_path, destpath])


def load_pickle(pickle_path: str) -> Any:
    """Load a pickle file."""
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    return data


def save_pickle(data: Any, pickle_path: str):
    """Save data in a pickle file."""
    with open(pickle_path, "wb") as f:
        pickle.dump(data, f, protocol=4)


def load_pth(pth_path: str) -> Any:
    """Load a pth (PyTorch) file."""
    data = torch.load(pth_path)
    return data


def save_pth(data: Any, pth_path: str):
    """Save data in a pth (PyTorch) file."""
    torch.save(data, pth_path)


def load_csv(csv_path: str, header: Any = None) -> pd.DataFrame:
    """Load a csv file."""
    try:
        data = pd.read_csv(csv_path, header=header)
    except pd.errors.EmptyDataError:
        data = pd.DataFrame()
    return data


def save_csv(data: Any, csv_path: str):
    """Save data in a csv file."""
    pd.DataFrame(data).to_csv(csv_path, header=False, index=False)


def write_clip(frames: List[np.array], output_filename: str, fps: float = 24):
    """Write a clip in `mp4` format from a list of frames.

    :param frames: RGB frames to write.
    :param output_filename: file name of the saved output.
    :param fps: wanted frame per second rate.
    """
    frame_height, frame_width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # Initialize the video writer
    clip = cv2.VideoWriter(
        output_filename,
        fourcc,
        fps,
        (frame_width, frame_height),
    )
    # Write each frame
    for frame in frames:
        clip.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # Release the video writer
    clip.release()


def load_frames_fromdir(video_dir: str) -> List[np.array]:
    """Load BGR frames from a directory of frames."""
    frames = []
    for frame_filename in sorted(os.listdir(video_dir)):
        frame_path = osp.join(video_dir, frame_filename)
        frames.append(cv2.imread(frame_path))

    return frames


def load_frames(video_path: str) -> List[np.array]:
    """Load BGR frames from a video file."""
    video_clip = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(int(video_clip.get(7))):
        _, frame = video_clip.read()
        if frame is None:
            break
        frames.append(frame)

    return frames
