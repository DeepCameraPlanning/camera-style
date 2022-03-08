import argparse
import os
import os.path as osp
from typing import Tuple

import torch

from src.utils.flow_utils import FlowUtils
from src.datamodules.preprocessing.flow_transforms import ResizeFlow

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
        "--frame-size",
        "-f",
        type=float,
        default=224,
        help="Resizing size",
    )

    args = parser.parse_args()

    return args.pth_dir, args.frame_size


def print_stats(stats: Stats, coordinates: str):
    """Display stats."""
    c1, c2 = ("x", "y") if coordinates == "xy" else ("r", "theta")
    mins, maxs, means, stds = stats
    print(f"min : {c1}={mins[0]:7.2f} - {c2}={mins[1]:7.2f} ")
    print(f"max : {c1}={maxs[0]:7.2f} - {c2}={maxs[1]:7.2f} ")
    print(f"mean: {c1}={means[0]:7.2f} - {c2}={means[1]:7.2f} ")
    print(f"std : {c1}={stds[0]:7.2f} - {c2}={stds[1]:7.2f} ")


def get_stats(xy_flows: torch.Tensor) -> Tuple[Stats]:
    """Compute xy and polar min/max/mean/std."""
    F = FlowUtils()
    polar_flows = torch.stack(
        [F.xy_to_polar(flow.permute([1, 2, 0])) for flow in xy_flows]
    )

    # Get stats in Euclidean coordinates
    x_flows, y_flows = xy_flows[:, 0], xy_flows[:, 1]
    xy_flows_mins = (x_flows.min().item(), y_flows.min().item())
    xy_flows_maxs = (x_flows.max().item(), y_flows.max().item())
    xy_flows_means = (x_flows.mean().item(), y_flows.mean().item())
    xy_flows_stds = (x_flows.mean().item(), y_flows.mean().item())
    xy_stats = xy_flows_mins, xy_flows_maxs, xy_flows_means, xy_flows_stds

    # Get stats in polar coordinates
    m_flows, a_flows = polar_flows[:, :, :, 0], polar_flows[:, :, :, 1]
    po_flows_mins = (m_flows.min().item(), a_flows.min().item())
    po_flows_maxs = (m_flows.max().item(), a_flows.max().item())
    po_flows_means = (m_flows.mean().item(), a_flows.mean().item())
    po_flows_stds = (m_flows.mean().item(), a_flows.mean().item())
    po_stats = po_flows_mins, po_flows_maxs, po_flows_means, po_flows_stds

    return xy_stats, po_stats


if __name__ == "__main__":
    flows = []
    pth_dir, frame_size = parse_arguments()
    resizer = ResizeFlow((frame_size, frame_size))
    for flow_dirname in os.listdir(pth_dir):
        flow_dirpath = osp.join(pth_dir, flow_dirname)
        for flow_filename in sorted(os.listdir(flow_dirpath)):
            flow_path = osp.join(flow_dirpath, flow_filename)
            flows.append(resizer(torch.load(flow_path)))
    flows = torch.stack(flows)

    xy_stats, po_stats = get_stats(flows)
    print("XY stats:")
    print_stats(xy_stats, "xy")
    print("\nPolar stats:")
    print_stats(po_stats, "polar")
