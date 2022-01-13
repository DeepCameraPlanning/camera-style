import os
import os.path as osp
import pickle
import subprocess
from typing import Any, Sequence

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
