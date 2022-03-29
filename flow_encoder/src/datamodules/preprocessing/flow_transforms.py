import argparse
import os
import os.path as osp
from typing import Tuple

from tqdm import tqdm
import torch
from torchvision.transforms import Compose, ToTensor
import torchvision.transforms.functional as F

from utils.file_utils import create_dir, load_pth, save_pth
from utils.flow_utils import FlowUtils


class ResizeFlow(torch.nn.Module):
    """
    Resize the input flow to the given size.
    Input/output shape: (H, W, C).
    """

    def __init__(
        self,
        size: Tuple[int, int],
        interpolation=F.InterpolationMode.BILINEAR,
        max_size=None,
        antialias=None,
    ):
        super().__init__()
        self.flow_utils = FlowUtils()

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            interpolation = F._interpolation_modes_from_int(interpolation)
        self.interpolation = interpolation
        self.antialias = antialias
        self.size = size
        self.max_size = max_size

    def forward(self, xy_flow: torch.Tensor) -> torch.Tensor:
        """Convert xy-flow to RGB, resize and re-convert to xy-flow."""
        rgb_flow = self.flow_utils.flow_to_frame(xy_flow.numpy(), method="hsv")
        rgb_flow = ToTensor()(rgb_flow)
        resized_rgb_flow = (
            F.resize(
                rgb_flow,
                self.size,
                self.interpolation,
                self.max_size,
                self.antialias,
            )
            .permute([1, 2, 0])
            .numpy()
        )
        resized_flow = self.flow_utils.frame_to_flow(resized_rgb_flow)

        return torch.Tensor(resized_flow)

    def __repr__(self):
        interpolate_str = self.interpolation.value
        return (
            self.__class__.__name__
            + f"(size={self.size}, interpolation={interpolate_str}, "
            + f"max_size={self.max_size}, antialias={self.antialias})"
        )


class ScaleFlow(torch.nn.Module):
    """
    Scale the input flow module.
    Input/output shape: (H, W, C).

    :param unit_module: wether to set all module to 1.
    :param max_module: value for scaling modules.
    """

    def __init__(self, unit_module: bool = False, max_module: float = None):
        super().__init__()
        self.flow_utils = FlowUtils()

        self.max_module = max_module
        self.unit_module = unit_module

    @staticmethod
    def _scale_module(
        flow: torch.Tensor, unit_module: bool = False, max_module: float = None
    ) -> torch.Tensor:
        """
        Scale module in polar coordinates between -1 and 1.
        Input shape (H, W, C)."""
        scaled_flow = torch.zeros_like(flow)
        scaled_flow[:, :, 1] = flow[:, :, 1]

        # Set all module to 1
        if unit_module:
            scaled_flow[:, :, 0] = torch.ones_like(scaled_flow[:, :, 0])
        # Scale modules by `max_mod`
        else:
            scaled_flow[:, :, 0] = flow[:, :, 0] / max_module

        return scaled_flow

    def forward(self, xy_flow: torch.Tensor) -> torch.Tensor:
        """Scale the input flow module."""
        polar_flow = self.flow_utils.xy_to_polar(xy_flow)
        scaled_polar_flow = self._scale_module(
            polar_flow,
            unit_module=self.unit_module,
            max_module=self.max_module,
        )
        scaled_xy_flow = self.flow_utils.polar_to_xy(scaled_polar_flow)

        return torch.Tensor(scaled_xy_flow)

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(max_module={self.max_module }, "
            + f"unit_module={self.unit_module})"
        )


def parse_arguments() -> Tuple[str, float, bool, int]:
    """Parse input arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "flow_dir",
        type=str,
        help="Path to the directory containing pth flows",
    )
    parser.add_argument(
        "save_dir",
        type=str,
        help="Path to the directory to save preprocessed flows",
    )
    parser.add_argument(
        "--max-module",
        "-m",
        type=float,
        default=None,
        help="Target size to resize frames",
    )
    parser.add_argument(
        "--unit-module",
        "-u",
        action="store_true",
        help="Wether to set all module to 1",
    )
    parser.add_argument(
        "--frame-size",
        "-f",
        type=int,
        default=224,
        help="Target size to resize frames",
    )
    args = parser.parse_args()

    return (
        args.flow_dir,
        args.save_dir,
        args.max_module,
        args.unit_module,
        args.frame_size,
    )


if __name__ == "__main__":
    flow_dir, save_dir, max_module, unit_module, frame_size = parse_arguments()

    transforms = Compose(
        [
            ResizeFlow((frame_size, frame_size)),
            ScaleFlow(unit_module, max_module),
        ]
    )
    for clip_dirname in tqdm(os.listdir(flow_dir)):
        clip_dir = osp.join(flow_dir, clip_dirname)
        save_clip_dir = osp.join(save_dir, clip_dirname)
        create_dir(save_clip_dir)
        for flow_filename in sorted(os.listdir(clip_dir)):
            flow_path = osp.join(clip_dir, flow_filename)
            save_flow_path = osp.join(save_clip_dir, flow_filename)
            flow = load_pth(flow_path)
            preprocessed_flow = transforms(flow)
            save_pth(preprocessed_flow, save_flow_path)
