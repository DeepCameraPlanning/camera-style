import os
import os.path as osp
import pickle
import subprocess
from typing import Any, List, Sequence, Tuple

import cv2
import librosa
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from pytorch_lightning.utilities import rank_zero_only
import rich.tree
import rich.syntax
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


def save_pth(data: Any, pth_path: str):
    """Save data in a pth (PyTorch) file."""
    torch.save(data, pth_path)


def load_pth(pth_path: str) -> Any:
    """Load a pth (PyTorch) file."""
    data = torch.load(pth_path)
    return data


def save_pickle(data: Any, pickle_path: str):
    """Save data in a pickle file."""
    with open(pickle_path, "wb") as f:
        pickle.dump(data, f)


def load_csv(csv_path: str, header: Any = None) -> pd.DataFrame:
    """Load a csv file."""
    data = pd.read_csv(csv_path, header=header)
    return data


def save_csv(data: Any, csv_path: str):
    """Save data in a csv file."""
    pd.DataFrame(data).to_csv(csv_path, header=False, index=False)


def load_frames_fromdir(video_dir: str) -> List[np.array]:
    """Load BGR frames from a directory of frames."""
    frames = []
    for frame_filename in os.listdir(video_dir):
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


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "compnode",
        "model",
        "datamodule",
        "xp_name",
        "seed",
    ),
    resolve: bool = True,
) -> None:
    """
    Adapted from: https://github.com/ashleve/lightning-hydra-template.
    Prints content of DictConfig using Rich library and its tree structure.

    :param config: configuration composed by Hydra.
    :param fields: determines which main fields from config will be printed and
        in what order.
    :param resolve: whether to resolve reference fields of DictConfig.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w") as fp:
        rich.print(tree, file=fp)


def save_spectrogram(
    spec: torch.Tensor,
    path: str,
    title: str = None,
    ylabel: str = "freq_bin",
    aspect: str = "auto",
    xmax: float = None,
):
    """Save a spectrogram plot."""
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(spec), origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.savefig(path)


def compute_bbox_area(bboxes: np.array) -> np.array:
    """Compute areas of a set of bounding-boxes."""
    width = bboxes[:, 2] - bboxes[:, 0]
    height = bboxes[:, 3] - bboxes[:, 1]

    return width * height


def compute_bbox_overlap(bboxes: np.array, bbox: np.array) -> np.array:
    """Compute overlaps between a set of bounding-boxes and one box."""
    xmin = np.maximum(bboxes[:, 0], bbox[0])
    xmax = np.minimum(bboxes[:, 2], bbox[2])
    width = np.maximum(0, xmax - xmin)

    ymin = np.maximum(bboxes[:, 1], bbox[1])
    ymax = np.minimum(bboxes[:, 3], bbox[3])
    height = np.maximum(0, ymax - ymin)

    return width * height


def compute_bbox_iou(bboxes: np.array, bbox: np.array) -> np.array:
    """Compute IoU between a set of bounding-boxes and one box."""
    areas_1 = compute_bbox_area(bboxes)
    areas_2 = compute_bbox_area(bbox.reshape(1, -1))

    overlaps = compute_bbox_overlap(bboxes, bbox)

    iou = overlaps / (areas_1 + areas_2 - overlaps)

    return iou


def write_clip(frames: List[np.array], output_filename: str, fps: float = 24):
    """Write a clip in `mp4` format from a list of frames.

    :param frames: 1st dim frame index, 2nd dim frame to be stacked
        together (must be the same dimensions).
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
        clip.write(frame)

    # Release the video writer
    clip.release()


def pad_frame(frame: np.array, grid_dims: Tuple[int, int]) -> np.array:
    """Pad frame to have dimensions divisble by `grid_dims`."""
    frame_height, frame_width = frame.shape[:2]
    n_row, n_col = grid_dims

    padding_height = (n_row - frame_height % n_row) % n_row
    # Check if `padding_height` is not divisible by 2
    if padding_height % 2:
        before_padding_height = padding_height // 2
        after_padding_height = padding_height // 2 + 1
    else:
        before_padding_height = padding_height // 2
        after_padding_height = padding_height // 2

    padding_width = (n_col - frame_width % n_col) % n_col
    # Check if `padding_width` is not divisible by 2
    if padding_width % 2:
        before_padding_width = padding_width // 2
        after_padding_width = padding_width // 2 + 1
    else:
        before_padding_width = padding_width // 2
        after_padding_width = padding_width // 2

    pad_dims = [
        (before_padding_height, after_padding_height),
        (before_padding_width, after_padding_width),
    ]
    if len(frame.shape) == 3:
        pad_dims.append((0, 0))

    padded_frame = np.pad(frame, pad_dims, mode="edge")

    return padded_frame


def get_patches(frame: np.array, grid_dims: Tuple[int, int]) -> np.array:
    """Split a frame into a grid of patches according to `grid_dims`."""
    padded_frame = pad_frame(frame, grid_dims)

    frame_height, frame_width = padded_frame.shape[:2]
    n_row, n_col = grid_dims
    y_stride, x_stride = frame_height // n_row, frame_width // n_col

    patches = [[None for _ in range(n_col)] for _ in range(n_row)]
    for i, y in enumerate(range(0, frame_height, y_stride)):
        for j, x in enumerate(range(0, frame_width, x_stride)):
            x_slice = slice(x, x + x_stride)
            y_slice = slice(y, y + y_stride)
            patches[i][j] = padded_frame[y_slice, x_slice]

    return np.array(patches)
