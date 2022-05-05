import argparse
import os
import os.path as osp
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from utils.flow_utils import FlowUtils
from flow_encoder.src.datamodules.preprocessing.flow_transforms import (
    ResizeFlow,
)

Stats = Tuple[
    Tuple[float, float],
    Tuple[float, float],
    Tuple[float, float],
    Tuple[float, float],
]


def parse_arguments() -> str:
    """Parse input arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "pth_dir",
        type=str,
        help="Path to the pth flow directory",
    )
    parser.add_argument(
        "hist_path",
        default=None,
        type=str,
        help="Path to save histograms",
    )
    parser.add_argument(
        "--frame-size",
        "-f",
        type=float,
        default=224,
        help="Resizing size",
    )

    args = parser.parse_args()

    return args.pth_dir, args.frame_size, args.hist_path


def print_stats(stats: Stats, coordinates: str):
    """Display stats."""
    c1, c2 = ("x", "y") if coordinates == "xy" else ("r", "theta")
    mins, maxs, means, stds, q1, q9 = stats
    print(f"min : {c1}={mins[0]:7.2f} - {c2}={mins[1]:7.2f} ")
    print(f"max : {c1}={maxs[0]:7.2f} - {c2}={maxs[1]:7.2f} ")
    print(f"mean: {c1}={means[0]:7.2f} - {c2}={means[1]:7.2f} ")
    print(f"std : {c1}={stds[0]:7.2f} - {c2}={stds[1]:7.2f} ")
    print(
        f"q1,9: {c1}=({q1[0]:.2f}, {q9[0]:.2f}) "
        + f"- {c2}=({q1[1]:.2f}, {q9[1]:.2f}) "
    )


def get_stats(xy_flows: torch.Tensor) -> Tuple[Stats]:
    """Compute xy and polar min/max/mean/std/quantiles."""
    F = FlowUtils()
    polar_flows = torch.stack([F.xy_to_polar(flow) for flow in xy_flows])

    # Get stats in Euclidean coordinates
    x_flows, y_flows = xy_flows[:, :, :, 0], xy_flows[:, :, :, 1]
    xy_flows_mins = (x_flows.min().item(), y_flows.min().item())
    xy_flows_maxs = (x_flows.max().item(), y_flows.max().item())
    xy_flows_means = (x_flows.mean().item(), y_flows.mean().item())
    xy_flows_stds = (x_flows.mean().item(), y_flows.mean().item())
    xy_flows_q1 = (np.quantile(x_flows, 0.1), np.quantile(y_flows, 0.1))
    xy_flows_q9 = (np.quantile(x_flows, 0.9), np.quantile(y_flows, 0.9))
    xy_stats = (
        xy_flows_mins,
        xy_flows_maxs,
        xy_flows_means,
        xy_flows_stds,
        xy_flows_q1,
        xy_flows_q9,
    )

    # Get stats in polar coordinates
    m_flows, a_flows = polar_flows[:, :, :, 0], polar_flows[:, :, :, 1]
    po_flows_mins = (m_flows.min().item(), a_flows.min().item())
    po_flows_maxs = (m_flows.max().item(), a_flows.max().item())
    po_flows_means = (m_flows.mean().item(), a_flows.mean().item())
    po_flows_stds = (m_flows.mean().item(), a_flows.mean().item())
    po_flows_q1 = (np.quantile(m_flows, 0.1), np.quantile(a_flows, 0.1))
    po_flows_q9 = (np.quantile(m_flows, 0.9), np.quantile(a_flows, 0.9))
    po_stats = (
        po_flows_mins,
        po_flows_maxs,
        po_flows_means,
        po_flows_stds,
        po_flows_q1,
        po_flows_q9,
    )

    return polar_flows, xy_stats, po_stats


if __name__ == "__main__":
    x_flows, x_norm_flows = [], []
    y_flows, y_norm_flows = [], []
    pth_dir, frame_size, hist_path = parse_arguments()

    n_elements = 0
    heights, widths = 0, 0
    F = FlowUtils()
    R = ResizeFlow((224, 224))
    xy_flows = []
    for flow_dirname in tqdm(os.listdir(pth_dir)):
        flow_dirpath = osp.join(pth_dir, flow_dirname)
        xy_flows.extend(
            torch.load(osp.join(flow_dirpath, flow_filename))
            for flow_filename in os.listdir(flow_dirpath)
        )
    xy_flows = torch.stack(xy_flows)
    polar_flows, xy_stats, po_stats = get_stats(xy_flows)
    print_stats(xy_stats, coordinates="xy")
    print_stats(po_stats, coordinates="po")

    if hist_path is not None:
        q9 = int(po_stats[5][0]) + 1
        plt.hist(
            polar_flows[:, :, :, 0].view(-1).numpy(),
            range=(0, q9),
            bins=np.arange(0, q9, q9 / 20),
        )
        plt.savefig(hist_path)
