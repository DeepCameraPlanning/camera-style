import logging
import os.path as osp
from typing import Tuple, List

import cv2
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lib.mannequinchallenge.models.hourglass import HourglassModel


class DepthModel(LightningModule):
    def __init__(self):
        super(DepthModel, self).__init__()
        self.model = HourglassModel(num_input=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logdepth_map, _ = self.model(x)
        return logdepth_map


class DepthEstimator:
    """
    Estimate depth given a frame.
    Code adapted from: https://github.com/google/mannequinchallenge.

    :param model_dir: path to model's directory.
    :param batch_size: how many samples per batch to load.
    :param num_gpus: how many gpus to use for inference.
    :param num_nodes: how many nodes to use for inference.
    :param num_workers: how many subprocesses to use for data loading.
    :param verbose: whether to display a progress bar or not.
    """

    def __init__(
        self,
        model_dir: str = "./models",
        batch_size: int = 5,
        num_gpus: int = 1,
        num_nodes: int = 1,
        num_workers: int = 12,
        verbose: bool = False,
    ):
        # Disable useless logs
        pl.utilities.distributed.log.setLevel(logging.ERROR)

        self.batch_size = batch_size
        self.num_workers = num_workers

        progress_bar_refresh_rate = None
        if not verbose:
            progress_bar_refresh_rate = 0

        self._checkpoint_filename = osp.join(model_dir, "depth_estimation.pth")
        self._state_dict = {
            "model." + parameter_name: parameters
            for parameter_name, parameters in torch.load(
                self._checkpoint_filename
            ).items()
        }
        model = DepthModel().eval()
        model.load_state_dict(self._state_dict)
        self.model = model
        self.trainer = Trainer(
            gpus=num_gpus,
            num_nodes=num_nodes,
            accelerator="dp",
            logger=False,
            progress_bar_refresh_rate=progress_bar_refresh_rate,
        )

    @staticmethod
    def _preprocess_frames(frames: List[np.array]) -> List[torch.Tensor]:
        """
        Pre-process a list of frames:
            - Resize;
            - Convert to RGB.
        """
        preprocesser = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize([288, 512]),
                transforms.ToTensor(),
            ]
        )
        processed_frames = [
            preprocesser(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            for frame in frames
        ]

        return processed_frames

    @staticmethod
    def _postprocess_maps(
        logdepth_maps: torch.Tensor, original_dims: Tuple[int, int]
    ) -> List[np.array]:
        """
        Post-process a list of log-depth maps:
            - Apply exponential;
            - Normalize;
            - Resize;
            - Move to CPU.
        """
        depth_maps = torch.exp(logdepth_maps)
        depth_maps /= torch.max(depth_maps)

        depth_maps = (
            F.interpolate(
                depth_maps,
                original_dims,
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(1)
            .cpu()
            .numpy()
        )

        return depth_maps

    def estimate_depth(self, frames: List[np.array]) -> List[np.array]:
        """
        Get depth estimation (1 is far, 0 is near) and a confidence map for
        each given frame.
        Code adapted from: https://github.com/google/mannequinchallenge.

        :param frames: list of BGR frames.
        :return: list of 0-1 ranged estimated depth maps.
        """
        preprocessed_frames = self._preprocess_frames(frames)
        frame_loader = DataLoader(
            preprocessed_frames,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        logdepth_maps = torch.vstack(
            self.trainer.predict(self.model, frame_loader)
        )

        original_dims = frames[0].shape[:2]
        depth_maps = self._postprocess_maps(logdepth_maps, original_dims)

        return depth_maps
