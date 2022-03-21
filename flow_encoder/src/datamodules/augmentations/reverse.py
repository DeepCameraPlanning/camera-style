import os
import os.path as osp
import subprocess

import torch

from src.utils.file_utils import create_dir


def reverse_clip(input_dir: str, save_dir: str):
    """Copy frames from `input_dir` in a reversed order in `save_dir`."""
    create_dir(save_dir)

    toward_files = sorted(os.listdir(input_dir))
    backward_files = toward_files[::-1]
    toward_to_backward = {t: b for t, b in zip(toward_files, backward_files)}
    for toward_filename in toward_files:
        toward_path = osp.join(input_dir, toward_filename)
        backward_path = osp.join(save_dir, toward_to_backward[toward_filename])
        subprocess.call(["cp", toward_path, backward_path])


def reverse_flow(input_dir: str, save_dir: str):
    """Copy neg flows from `input_dir` in a reversed order in `save_dir`."""
    create_dir(save_dir)

    toward_files = sorted(os.listdir(input_dir))
    backward_files = toward_files[::-1]
    toward_to_backward = {t: b for t, b in zip(toward_files, backward_files)}
    for toward_filename in toward_files:
        toward_path = osp.join(input_dir, toward_filename)
        backward_path = osp.join(save_dir, toward_to_backward[toward_filename])
        torward_flow = torch.load(toward_path)
        torch.save(-torward_flow, backward_path)
