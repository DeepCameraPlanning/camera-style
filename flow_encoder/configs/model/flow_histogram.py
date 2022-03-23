from typing import Tuple

import torch
import torch.nn as nn


class FlowHistogram(nn.Module):
    """Flow histogram extractor."""

    def __init__(self, grid_dims: Tuple[int, int], n_angle_bins: int):
        """ """
        self._n_row, self._n_col = grid_dims
        self._n_angle_bins = n_angle_bins

    def extract_features(self, flow: torch.Tensor):
        """flow: (B, C, T, W, H)"""
        flow_height, flow_width = flow.shape[-2:]
        y_stride = flow_height // self._n_row + 1
        x_stride = flow_width // self._n_col + 1

        angle_map = [
            [None for _ in range(self._n_col)] for _ in range(self._n_row)
        ]
        for i, y in enumerate(range(0, flow_height, y_stride)):
            for j, x in enumerate(range(0, flow_width, x_stride)):
                # Extract the subflow
                x_slice = slice(x, x + x_stride)
                y_slice = slice(y, y + y_stride)
                sub_flow = flow[:, :, :, y_slice, x_slice]
                x_subflow = sub_flow[:, 0]
                y_subflow = sub_flow[:, 1]

                # Get angles
                flow_angles = torch.atan2(y_subflow, x_subflow)
                mean_flow_angles = flow_angles.mean(axis=(2, 3))

                # Quantize mean angle
                quantized_angle = torch.floor(
                    self._n_angle_bins
                    * (
                        torch.pi
                        + (torch.pi / self._n_angle_bins)
                        + mean_flow_angles
                    )
                    / (2 * torch.pi)
                )

                quantized_angle = torch.where(
                    quantized_angle == self._n_angle_bins,
                    torch.zeros_like(quantized_angle),
                    quantized_angle,
                )
                quantized_angle = torch.where(
                    quantized_angle < 0,
                    -torch.ones_like(quantized_angle),
                    quantized_angle,
                )
                angle_map[i][j] = quantized_angle

        angle_map = torch.stack([torch.stack(a) for a in angle_map])

        return angle_map.permute([2, 3, 0, 1])


def make_flow_histogram(
    grid_dims: Tuple[int, int], n_angle_bins: int
) -> nn.Module:
    """Flow histogram extractor."""
    model = FlowHistogram(grid_dims=grid_dims, n_angle_bins=n_angle_bins)
    return model
