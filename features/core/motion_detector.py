import logging
import argparse
import os
import os.path as osp
import sys
from typing import List, Tuple

import cv2
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from lib.motion_detection.estimators.model import MLP

# Avoid local import issues
root_dir = [os.sep] + osp.dirname(osp.abspath(__file__)).split(os.sep)[:-2]
import_dir = ["lib", "RAFT", "core"]
sys.path.append(osp.join(*root_dir + import_dir))
from raft import RAFT
from utils.flow_viz import flow_to_image
from utils.utils import InputPadder


class RAFTModel(LightningModule):
    def __init__(self):
        super(RAFTModel, self).__init__()
        sys.argv = [""]
        parser = argparse.ArgumentParser()
        parser.add_argument("--small", action="store_true")
        parser.add_argument("--mixed_precision", action="store_true")
        parser.add_argument("--alternate_corr", action="store_true")
        args = parser.parse_args()
        self.model = RAFT(args)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x1, x2 = x
        _, flow = self.model(x1, x2, iters=20, test_mode=True)
        return flow


class FlowEstimator:
    """
    Estimate flow given two frames.
    Code adapted from: https://github.com/princeton-vl/RAFT.

    :param model_name: name of the OF to use (`raft` or `tvl1`).
    :param model_dir: path to model's directory.
    :param batch_size: how many samples per batch to load.
    :param num_gpus: how many gpus to use for inference.
    :param num_nodes: how many nodes to use for inference.
    :param num_workers: how many subprocesses to use for data loading.
    :param verbose: whether to display a progress bar or not.
    """

    def __init__(
        self,
        model_name: str = "tvl1",
        model_dir: str = "./models",
        batch_size: int = 1,
        num_gpus: int = 1,
        num_nodes: int = 1,
        num_workers: int = 12,
        verbose: bool = False,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers

        enable_progress_bar = True
        if not verbose:
            pl.utilities.distributed.log.setLevel(logging.ERROR)
            enable_progress_bar = False

        self.model_name = model_name
        if self.model_name == "raft":
            self._checkpoint_filename = osp.join(model_dir, "raft-things.pth")
            self._state_dict = {
                "model." + parameter_name[7:]: parameters
                for parameter_name, parameters in torch.load(
                    self._checkpoint_filename
                ).items()
            }
            model = RAFTModel().eval()
            model.load_state_dict(self._state_dict)
            self.model = model
            self.trainer = Trainer(
                gpus=num_gpus,
                num_nodes=num_nodes,
                # strategy="dp",
                logger=False,
                enable_progress_bar=enable_progress_bar,
            )
        elif self.model_name == "tvl1":
            self.model = cv2.optflow.DualTVL1OpticalFlow_create()

    @staticmethod
    def _raft_preprocess_frames(frames: List[np.array]) -> List[torch.Tensor]:
        """
        Pre-process a list of RGB frames:
            # - Pad in order to have dimensions divisible by 8;
            # - Resize;
        """
        height, width = frames[0].shape[:2]
        preprocesser = transforms.Compose(
            [
                transforms.ToPILImage(),
                # transforms.Resize((height // 2, width // 2)),
                transforms.ToTensor(),
            ]
        )
        processed_frames = torch.stack(
            [preprocesser(frame) * 255 for frame in frames]
        )
        padder = InputPadder(processed_frames[0].shape)
        processed_frames = padder.pad(processed_frames)[0]

        return processed_frames

    @staticmethod
    def _raft_postprocess_flows(
        flows: torch.Tensor, original_dims: Tuple[int, int]
    ) -> List[np.array]:
        """Resize a list of optical flow estimations to original frame dims."""
        flows = transforms.Resize(original_dims)(flows).cpu()
        flows = flows.permute([0, 2, 3, 1]).numpy()
        return flows

    def estimate_flow(self, frames: List[np.array]) -> List[np.array]:
        """
        Estimate flow between two frames given a list of consecutive frames.
        Code adapted from: https://github.com/princeton-vl/RAFT.

        :param frames: list of RGB frames.
        :return: list estimated flows.
        """
        if self.model_name == "raft":
            preprocessed_frames = self._raft_preprocess_frames(frames)
            frame_loader = DataLoader(
                list(zip(preprocessed_frames[:-1], preprocessed_frames[1:])),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )
            flows = torch.vstack(
                self.trainer.predict(self.model, frame_loader)
            )

            original_dims = frames[0].shape[:2]
            flows = self._raft_postprocess_flows(flows, original_dims)

        elif self.model_name == "tvl1":
            gray_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
            flows = [
                self.model.calc(prev, curr, None)
                for prev, curr in zip(gray_frames[:-1], gray_frames[1:])
            ]

        return flows

    def flow_to_frame(self, flow: np.array) -> np.array:
        """Convert the output of the model to a RGB frame."""
        flowed_frame = flow_to_image(flow, convert_to_bgr=False)
        return flowed_frame


class MotionModel(LightningModule):
    def __init__(self):
        super(MotionModel, self).__init__()
        self.model = MLP()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        motion_detection = self.model(x)
        return motion_detection


class MotionDetector:
    """
    Detect camera motion given optical flows.

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
        batch_size: int = 64,
        num_gpus: int = 1,
        num_nodes: int = 1,
        num_workers: int = 12,
        verbose: bool = False,
    ):
        # Disable useless logs
        pl.utilities.distributed.log.setLevel(logging.ERROR)

        self.batch_size = batch_size
        self.num_workers = num_workers

        enable_progress_bar = True
        if not verbose:
            enable_progress_bar = False

        self._checkpoint_filename = osp.join(model_dir, "motion.pth")
        self._state_dict = {
            "model." + parameter_name: parameters
            for parameter_name, parameters in torch.load(
                self._checkpoint_filename
            ).items()
        }
        model = MotionModel().eval()
        model.load_state_dict(self._state_dict)
        self.model = model
        self.trainer = Trainer(
            gpus=num_gpus,
            num_nodes=num_nodes,
            # strategy="ddp",
            logger=False,
            enable_progress_bar=enable_progress_bar,
        )

        self.encoding_to_motion = {
            0: "x_translation",
            1: "y_translation",
            2: "z_rotation",
            3: "z_translation",
            4: "null",
        }

    @staticmethod
    def _get_anglemap(
        flow: np.array,
        grid_dims: Tuple[int, int],
        n_angle_bins: int = 8,
        magnitude_threshold: float = 0,
    ) -> List[np.array]:
        """
        Get the average flow angle for each block of a grid and quantize it.

        :param flows: optical flow to process (with x and y components).
        :param grid_dims: number of rows and columns of the grid.
        :param n_angle_bins: number of quantization bins for angles.
        :param magnitude_threshold: ignore angle if the magnitude is too low.
        :return: grid with average flow angle for each transition.
        """
        flow_height, flow_width = flow.shape[:2]
        n_row, n_col = grid_dims
        y_stride, x_stride = flow_height // n_row + 1, flow_width // n_col + 1

        angle_map = [[None for _ in range(n_col)] for _ in range(n_row)]
        for i, y in enumerate(range(0, flow_height, y_stride)):
            for j, x in enumerate(range(0, flow_width, x_stride)):
                # Extract the subflow
                x_slice = slice(x, x + x_stride)
                y_slice = slice(y, y + y_stride)
                sub_flow = flow[y_slice, x_slice]
                x_subflow = sub_flow[:, :, 0]
                y_subflow = sub_flow[:, :, 1]
                # Compute angles and magnitudes
                flow_angle = np.arctan2(y_subflow, x_subflow)
                flow_magnitude = np.sqrt(x_subflow**2 + y_subflow**2)
                # Filter low magnitude angles and average angles
                angle = (
                    flow_angle.mean()
                    if flow_magnitude.mean() > magnitude_threshold
                    else -10
                )
                # Quantize mean angle
                quantized_angle = int(
                    np.floor(
                        n_angle_bins
                        * (np.pi + (np.pi / n_angle_bins) + np.array(angle))
                        / (2 * np.pi)
                    )
                )
                if quantized_angle == n_angle_bins:
                    quantized_angle = 0
                if quantized_angle < 0:
                    quantized_angle = -1
                # # One-hot encode quantized angle
                # encoded_angle = np.eye(n_angle_bins + 1)[quantized_angle + 1]

                # angle_map[i][j] = encoded_angle
                angle_map[i][j] = quantized_angle
        return np.array(angle_map)

    @staticmethod
    def _preprocess_maps(
        forward_angle_maps: List[np.array],
        backward_angle_maps: List[np.array],
        window_size: int,
    ) -> List[torch.Tensor]:
        """
        Select forward and backward angle maps within a sliding window and
        concatenate them.
        """
        n_frames = len(forward_angle_maps) - window_size
        if n_frames < 1:
            raise ("Not enough frames")

        processed_maps = torch.stack(
            [
                torch.from_numpy(
                    np.hstack(
                        forward_angle_maps[slice(k, k + window_size)]
                        + backward_angle_maps[slice(k, k + window_size)]
                    ).astype(np.float32)
                ).reshape(-1)
                for k in range(n_frames)
            ]
        )

        return processed_maps

    @staticmethod
    def _postprocess_motions(
        motion_probas: torch.Tensor, window_size: int
    ) -> List[np.array]:
        """Post-process a list of motion detections."""
        # Assign detection with the most probable class
        detected_motions = torch.argmax(motion_probas, axis=1).cpu().tolist()

        # Use the first and last predictions for the first and last frames
        detected_motions = (
            [detected_motions[0]] * (window_size // 2)
            + detected_motions
            + [detected_motions[-1]] * (window_size // 2)
        )

        return detected_motions

    def detect_motion(
        self,
        forward_flows: np.array,
        backward_flows: np.array,
        window_size: int = 8,
    ) -> List[np.array]:
        """
        Infers the model and returns a motion class for each frame.

        :param forward_flows: list forward optical flow estimations.
        :param backward_flows: list backforward optical flow estimations.
        :param window_size: number of consecutive frames to take into account.
        :return: list of estimated defocus maps.
        """
        # Compute angle maps
        forward_angle_maps = [
            self._get_anglemap(flow, (6, 10)).reshape(-1)
            for flow in forward_flows
        ]
        backward_angle_maps = [
            self._get_anglemap(flow, (6, 10)).reshape(-1)
            for flow in backward_flows
        ]

        # Detect camera movements
        preprocessed_maps = self._preprocess_maps(
            forward_angle_maps, backward_angle_maps, window_size
        )
        map_loader = DataLoader(
            preprocessed_maps,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        motion_probas = torch.vstack(
            self.trainer.predict(self.model, map_loader)
        )
        detected_motions = self._postprocess_motions(
            motion_probas, window_size
        )

        return detected_motions

    @staticmethod
    def get_motionfeatures(motion_detections: List[int]) -> List[np.array]:
        """Compute motion features: one-hot encode motion detections."""
        motion_features = np.eye(5)[motion_detections]
        return motion_features
