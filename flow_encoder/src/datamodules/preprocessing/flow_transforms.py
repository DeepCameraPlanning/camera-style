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


def generalized_logistic(
    x: torch.Tensor,
    B: torch.Tensor,
    Q: torch.Tensor,
    A: torch.Tensor = 0,
    K: torch.Tensor = 1,
    C: torch.Tensor = 1,
):
    return A + ((K - A) / (C + Q * torch.exp(-B * torch.tensor(x))))


# RAFT
# B = torch.tensor(0.6090)
# Q = torch.tensor(25.2964)


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
    :param scale_module: value for scaling modules.
    :param step_module: value for scaling modules.
    :param quantile: quantile values (q1, q9) for scaling modules.
    """

    def __init__(
        self,
        unit_module: bool = False,
        scale_module: float = None,
        step_module: float = None,
        quantiles: Tuple[float, float] = None,
    ):
        super().__init__()
        self.flow_utils = FlowUtils()

        self.scale_module = scale_module
        self.unit_module = unit_module
        self.step_module = step_module
        self.quantiles = quantiles

    @staticmethod
    def _scale_module(
        flow: torch.Tensor,
        unit_module: bool = False,
        scale_module: float = None,
        step_module: float = None,
        quantiles: float = None,
    ) -> torch.Tensor:
        """
        Scale module in polar coordinates between -1 and 1.
        Input shape (H, W, C).
        """
        scaled_flow = torch.zeros_like(flow)
        scaled_flow[:, :, 1] = flow[:, :, 1]

        # Set all module to 1
        if unit_module:
            scaled_flow[:, :, 0] = torch.ones_like(scaled_flow[:, :, 0])
        # Scale modules by `scale_module`
        elif scale_module is not None:
            scaled_flow[:, :, 0] = torch.clip(
                flow[:, :, 0] / scale_module, min=0, max=1
            )
        elif step_module is not None:
            scaled_flow[:, :, 0] = 1 * (flow[:, :, 0] > step_module)
        # Scale modules with a logistic
        else:
            q1, q9 = quantiles
            a = 1 / 0.05
            b = 1 / 0.95
            B = torch.log(torch.tensor((a - 1) / (b - 1))) / (q9 - q1)
            Q = (a - 1) * torch.exp(B * q1)
            scaled_flow[:, :, 0] = generalized_logistic(flow[:, :, 0], B, Q)

        return scaled_flow

    def forward(self, xy_flow: torch.Tensor) -> torch.Tensor:
        """Scale the input flow module."""
        polar_flow = self.flow_utils.xy_to_polar(xy_flow)

        scaled_polar_flow = self._scale_module(
            polar_flow,
            unit_module=self.unit_module,
            scale_module=self.scale_module,
            step_module=self.step_module,
            quantiles=self.quantiles,
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
        "--scale-module",
        "-sc",
        type=float,
        default=None,
        help="Target size to resize frames",
    )
    parser.add_argument(
        "--step-module",
        "-st",
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
        "-q1",
        type=float,
        default=None,
        help="Wether to logistic normalize with q1",
    )
    parser.add_argument(
        "-q9",
        type=float,
        default=None,
        help="Wether to logistic normalize with q9",
    )

    args = parser.parse_args()

    return (
        args.flow_dir,
        args.save_dir,
        args.scale_module,
        args.unit_module,
        args.step_module,
        args.q1,
        args.q9,
    )


if __name__ == "__main__":
    (
        flow_dir,
        save_dir,
        scale_module,
        unit_module,
        step_module,
        q1,
        q9,
    ) = parse_arguments()

    transforms = Compose(
        [
            ScaleFlow(unit_module, scale_module, step_module, (q1, q9)),
            ResizeFlow((224, 224)),
        ]
    )
    for clip_dirname in tqdm(os.listdir(flow_dir)):
        clip_dir = osp.join(flow_dir, clip_dirname)
        save_clip_dir = osp.join(save_dir, clip_dirname)
        create_dir(save_clip_dir)
        for flow_filename in sorted(os.listdir(clip_dir)):
            flow_path = osp.join(clip_dir, flow_filename)
            save_flow_path = osp.join(save_clip_dir, flow_filename)
            flow = load_pth(flow_path).float()
            preprocessed_flow = transforms(flow)
            save_pth(preprocessed_flow, save_flow_path)
