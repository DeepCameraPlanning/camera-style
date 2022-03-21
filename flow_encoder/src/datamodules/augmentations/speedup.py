import os
import os.path as osp
import subprocess

import torch
import numpy as np

from src.utils.file_utils import create_dir


def speedup_clip(input_dir: str, save_dir: str):
    """Copy frames from `input_dir`, drop 1 frame over 2 in `save_dir`."""
    create_dir(save_dir)

    flow_files = sorted(os.listdir(input_dir))
    dropped_files = np.array(flow_files)[np.arange(0, len(flow_files), 2)]
    for file_index, flow_filename in enumerate(dropped_files):
        flow_path = osp.join(input_dir, flow_filename)
        dropped_path = osp.join(save_dir, str(file_index).zfill(4) + ".png")
        subprocess.call(["cp", flow_path, dropped_path])


def speedup_flow(input_dir: str, save_dir: str):
    """
    Copy frames from `input_dir`, add dropped frames to the following in
    `save_dir`.
    """
    create_dir(save_dir)

    flow_files = sorted(os.listdir(input_dir))
    dropped_files = np.array(flow_files)[np.arange(1, len(flow_files), 2)]
    for dropped_index, last_flow_filename in enumerate(dropped_files):
        inter_flow_index = int(last_flow_filename[:-4]) - 1
        inter_flow_filename = str(inter_flow_index).zfill(4) + ".pth"

        last_flow_path = osp.join(input_dir, last_flow_filename)
        inter_flow_path = osp.join(input_dir, inter_flow_filename)
        dropped_path = osp.join(save_dir, str(dropped_index).zfill(4) + ".pth")

        last_flow = torch.load(last_flow_path)
        inter_flow = torch.load(inter_flow_path)
        torch.save(inter_flow + last_flow, dropped_path)
