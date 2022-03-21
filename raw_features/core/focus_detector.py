from itertools import product
import logging
import os
import os.path as osp
import sys
from typing import List, Tuple
import warnings

import cv2
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from movie_style.tools.utils import get_patches

# Avoid local import issues
root_dir = [os.sep] + osp.dirname(osp.abspath(__file__)).split(os.sep)[:-2]
import_dir = ["lib", "depth_distillation"]
sys.path.append(osp.join(*root_dir + import_dir))
from models.fcn import ResNextDecoderAtt
from models.vgg import VGGNet


class FocusModel(LightningModule):
    def __init__(self):
        super(FocusModel, self).__init__()
        self.model = ResNextDecoderAtt(pretrained_net=VGGNet(), type="vgg")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            focus_map = self.model(x)[0][0]

        return focus_map


class FocusDetector:
    """
    Detect defocus blur given a frame.
    Code adapted from: https://github.com/vinthony/depth-distillation.

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
        batch_size: int = 25,
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

        self._checkpoint_filename = osp.join(
            model_dir, "defocus_estimation_vgg.pth"
        )
        self._state_dict = {
            "model." + parameter_name: parameters
            for parameter_name, parameters in torch.load(
                self._checkpoint_filename
            ).items()
        }
        model = FocusModel().eval()
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
            - Normalize;
            - Convert to RGB.
        """
        preprocesser = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(
                    (320, 320),
                    transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )

        processed_frames = [
            preprocesser(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            for frame in frames
        ]

        return processed_frames

    @staticmethod
    def _postprocess_maps(
        focus_maps: torch.Tensor, original_dims: Tuple[int, int]
    ) -> List[np.array]:
        """
        Post-process a list of focus maps:
            - Normalize;
            - Resize;
            - Move to CPU.
        """
        focus_maps = (
            F.interpolate(
                torch.sigmoid(focus_maps),
                original_dims,
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(1)
            .cpu()
            .numpy()
        )

        return focus_maps

    def detect_focus(self, frames: List[np.array]) -> List[np.array]:
        """
        Infers the model and returns normalized focus map (1 is out-of-focus
        and 0 is in-focus).
        Code adapted from: https://github.com/vinthony/depth-distillation.

        :param frames: list of BGR frames.
        :return: list of 0-1 ranged estimated focus maps.
        """
        preprocessed_frames = self._preprocess_frames(frames)
        frame_loader = DataLoader(
            preprocessed_frames,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        focus_maps = torch.vstack(
            self.trainer.predict(self.model, frame_loader)
        )

        original_dims = frames[0].shape[:2]
        focus_maps = self._postprocess_maps(focus_maps, original_dims)

        return focus_maps

    @staticmethod
    def get_focusfeatures(
        focus_maps: List[np.array],
        featuremap_dims: Tuple[int, int],
        focus_threshold: float = 0.5,
    ) -> np.array:
        """Compute focus features: binarized focus grids.

        :param frames: frames to process.
        :param featuremap_dims: dimensions of the output focus grid.
        :param focus_threshold: threshold to binarize focus map.
        :return: binarized focus grids.
        """
        n_row, n_col = featuremap_dims
        focus_features = []
        for focus_map in focus_maps:
            focus_patches = get_patches(focus_map, (n_row, n_col))

            focus_grid = -np.ones((n_row, n_col))
            for i, j in product(np.arange(n_row), np.arange(n_col)):
                patch = focus_patches[i][j]
                # Binarize the patch
                binary_patch = (patch >= focus_threshold).reshape(-1)
                # Find the most frequent byte (0 or 1) and store it
                focus_grid[i][j] = np.bincount(binary_patch).argmax()

            focus_features.append(focus_grid)

        return focus_features
