from itertools import combinations
import logging
import os.path as osp
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from lib.dope.model import dope_resnet50
from lib.dope.postprocess import assign_hands_and_head_to_body
from lib.lcrnet_v2_improved_ppi.lcr_net_ppi_improved import LCRNet_PPI_improved
from movie_style.tools.utils import compute_bbox_iou


class PoseModel(LightningModule):
    def __init__(self, kwargs):
        super(PoseModel, self).__init__()
        self.model = dope_resnet50(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        poselets = self.model(x, None)
        return poselets


class PoseEstimator:
    """
    Estimate poselets given a frame.
    Code adapted from: https://github.com/naver/dope.

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
            model_dir, "DOPErealtime_v1_0_0.pth.tgz"
        )
        self._checkpoint = torch.load(self._checkpoint_filename)
        self._state_dict = {
            "model." + parameter_name: parameters
            for parameter_name, parameters in self._checkpoint[
                "state_dict"
            ].items()
        }
        model = PoseModel(self._checkpoint["dope_kwargs"]).eval().half()
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
        """Pre-process a list of frames."""
        processed_frames = [ToTensor()(frame).half() for frame in frames]
        return processed_frames

    def _process_output(
        self,
        frame_output: Dict[str, torch.Tensor],
        resolution: Tuple[int, int],
    ) -> Tuple[
        Dict[str, List[Dict[str, np.array]]],
        List[Tuple[int, int]],
        List[Tuple[int, int]],
    ]:
        """
        Process pose estimator output by integrating the pose proposals.
        Code adapted from: https://github.com/naver/dope.

        :param frame_output: all pose proposals estimate for a frame.
        :param resolution: resolution of frames.
        :return: processed poselets, linked body and wrists indices
            and linked body and heads indices
        """
        # Move output to cpu
        frame_output_cpu = {
            k: v.float().data.cpu().numpy() for k, v in frame_output.items()
        }
        detections = {}
        parts = ["body", "hand", "face"]
        for part in parts:
            detections[part] = LCRNet_PPI_improved(
                frame_output_cpu[part + "_scores"],
                frame_output_cpu["boxes"],
                frame_output_cpu[part + "_pose2d"],
                frame_output_cpu[part + "_pose3d"],
                resolution,
                **self._checkpoint[part + "_ppi_kwargs"]
            )

        # Assignment of hands and head to body
        (
            detections,
            body_with_wrists,
            body_with_head,
        ) = assign_hands_and_head_to_body(detections)

        return detections, body_with_wrists, body_with_head

    def _postprocess_poselets(
        self, raw_poselets: torch.Tensor, original_dims: Tuple[int, int]
    ) -> List[np.array]:
        """Post-process a list of raw poselets."""
        processed_poselets = []
        for batch_poselets in raw_poselets:
            for frame_poselets in batch_poselets:
                detections, _, _ = self._process_output(
                    frame_poselets, original_dims
                )
                # Get only body poselets
                poselets = [body["pose2d"] for body in detections["body"]]
                processed_poselets.append(np.array(poselets))

        return processed_poselets

    def estimate_pose(self, frames: List[np.array]) -> List[np.array]:
        """
        Get pose estimations of each given frame with DOPE inferences.
        For body dection, output indices correspond to:
            - 0: right ankle
            - 1: left ankle
            - 2: right knee
            - 3: left knee
            - 4: right hip
            - 5: left hip
            - 6: right wrist
            - 7: left wrist
            - 8: right elbow
            - 9: left elbow
            - 10: right shoulder
            - 11: left shoulder
            - 12: head

        Code adapted from: https://github.com/naver/dope.
        """
        preprocessed_frames = self._preprocess_frames(frames)
        frame_loader = DataLoader(
            preprocessed_frames,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        poselets = self.trainer.predict(self.model, frame_loader)

        original_dims = frames[0].shape[:2]
        poselets = self._postprocess_poselets(poselets, original_dims)

        return poselets

    def track_poselets(
        self,
        poselets: List[np.array],
        bbox_tracks: List[np.array],
        frame_dims: Tuple[int, int],
    ) -> List[List[np.array]]:
        """Detect poselets from a list of frames and track them.

        :param poselets: detected poselets for each frame.
        :param bbox_tracks: tracked bounding-boxes.
        :param frame_dims: frame dimensions to avoid out of the frame bbox.
        :param iou_threshold: minimum overlap between two bounding-boxes.
        :return: tracked poselets.
        """
        raw_poselet_tracks = self._get_tracks(
            poselets, bbox_tracks, frame_dims, self.iou_threshold
        )
        poselet_tracks = self._process_tracks(poselets, raw_poselet_tracks)

        return poselet_tracks

    @staticmethod
    def _process_tracks(
        poselets: List[np.array], raw_poselet_tracks: List[np.array]
    ) -> List[np.array]:
        """
        Process tracks:
            - Remove short tracks.
            - Interpolate missing frames.
        """

        def interpolate_poses(
            track: List[np.array], first_frame: int, last_frame: int
        ) -> bool:
            """Interpolate missing 2D poses."""
            active_track = [[k] for k in range(first_frame, last_frame + 1)]
            for pose_index in range(len(track)):
                frame_index = int(track[pose_index][0])
                pose = track[pose_index][1]
                # Update active track
                active_track[frame_index - first_frame] += [pose]

                # Check if there is a gap between the last and current frame
                if (pose_index > 0) and (
                    active_track[pose_index - 1][0] != frame_index - 1
                ):
                    # Interpolate missing frame
                    last_index = int(track[pose_index - 1][0])
                    last_pose = track[pose_index - 1][1]
                    for missing_index in range(last_index + 1, frame_index):
                        alpha = (frame_index - missing_index) / (
                            frame_index - last_index
                        )
                        active_track[missing_index - first_frame] += [
                            alpha * last_pose + (1 - alpha) * pose
                        ]

            return np.array(active_track, dtype=object)

        poselet_tracks = []
        for raw_track in raw_poselet_tracks:
            # Check if the track is empty or not (full of -1)
            if not (raw_track > 0).sum():
                continue

            # Remove useless frames from the track and get joints' coordinates
            processed_track = []
            for frame_index, pose_index in enumerate(raw_track):
                if pose_index >= 0:
                    pose = poselets[frame_index][int(pose_index)]
                    processed_track.append([frame_index, pose])

            processed_track = interpolate_poses(
                processed_track,
                processed_track[0][0],
                processed_track[-1][0],
            )
            poselet_tracks.append(processed_track)

        return poselet_tracks

    @staticmethod
    def _get_tracks(
        poselets: List[np.array],
        bbox_tracks: List[np.array],
        frame_dims: Tuple[int, int],
        iou_threshold: float = 0.4,
    ) -> List[np.array]:
        """Compare pose bounding-boxes with pre-tracked bounding-boxes."""
        height, width = frame_dims
        raw_poselet_tracks = -np.ones((len(bbox_tracks), len(poselets)))
        for frame_index in range(len(poselets)):
            frame_bboxes = []
            for track_index, track in enumerate(bbox_tracks):
                bbox = track[track[:, 0] == frame_index]
                if bbox.size:
                    frame_bboxes.append([track_index] + bbox[0, 1:].tolist())

            frame_bboxes = np.array(frame_bboxes)
            for poselet_index in range(len(poselets[frame_index])):
                if frame_bboxes.size:
                    current_poselet = poselets[frame_index][poselet_index]
                    x_min = np.max(
                        [np.min([np.min(current_poselet[:, 0]), width]), 0]
                    )
                    y_min = np.max(
                        [np.min([np.min(current_poselet[:, 1]), height]), 0]
                    )
                    x_max = np.max(
                        [np.min([np.max(current_poselet[:, 0]), width]), 0]
                    )
                    y_max = np.max(
                        [np.min([np.max(current_poselet[:, 1]), height]), 0]
                    )
                    poselet_bbox = np.array([x_min, y_min, x_max, y_max])

                    bbox_ious = compute_bbox_iou(
                        frame_bboxes[:, 1:], poselet_bbox
                    )
                    best_iou_index = np.argmax(bbox_ious)

                    if bbox_ious[best_iou_index] > iou_threshold:
                        track_index = int(frame_bboxes[best_iou_index][0])

                        raw_poselet_tracks[
                            track_index, frame_index
                        ] = poselet_index
                        frame_bboxes = np.delete(
                            frame_bboxes, best_iou_index, axis=0
                        )

        return raw_poselet_tracks

    def process_combinations(
        self, poselet_tracks: List[np.array], frame_dims: Tuple[int, int]
    ) -> Tuple[Tuple[np.array, np.array], Tuple[int, int], Tuple[int, int]]:
        """
        Format poselet as the input of the toric estimation model, and find
        all combinations of poselet along the frames.

        :param poselet_tracks: tracked poselets.
        :param frame_dims: frame dimensions to normalize poselets.
        :return: all combinations of 2 of normalized tracked poselets,
            corresponding track indices and corresponding frame indices.
        """

        def format_poselet(
            poselet: np.array, frame_dims: Tuple[int, int]
        ) -> Dict[str, np.array]:
            """
            Format poselet as the inputs of toric cordinate estimation
            model is described https://github.com/jianghd1996/Camera-control.

            ie:
                - head_joint
                - neck_joint
                - left_shoulder_joint
                - right_shoulder_joint
                - pelvis_joint
                - left_hip_joint
                - right_hip_joint
            And normalize it with frame width and height.
            """
            (
                head_joint,
                left_shoulder_joint,
                right_shoulder_joint,
                left_hip_joint,
                right_hip_joint,
            ) = poselet[[12, 11, 10, 4, 5]]
            neck_joint = (left_shoulder_joint + right_shoulder_joint) / 2
            pelvis_joint = (left_hip_joint + right_hip_joint) / 2

            relevant_joints = np.concatenate(
                [
                    head_joint,
                    neck_joint,
                    left_shoulder_joint,
                    right_shoulder_joint,
                    pelvis_joint,
                    left_hip_joint,
                    right_hip_joint,
                ]
            )

            height, width = frame_dims
            even_indices = [k for k in range(14) if k % 2 == 0]
            odd_indices = [k for k in range(14) if k % 2 != 0]
            relevant_joints[even_indices] /= width
            relevant_joints[odd_indices] /= height

            return relevant_joints

        # Format all poselet in order to be input of toric estimation model
        formatted_poselet_tracks = []
        for track in poselet_tracks:
            formatted_track = []
            for pose in track:
                frame_index, poselet = pose
                formatted_poselet = format_poselet(poselet, frame_dims)
                formatted_track.append([frame_index, formatted_poselet])
            formatted_poselet_tracks.append(formatted_track)

        combined_tracks = []
        for track_index_1, track_index_2 in combinations(
            range(len(formatted_poselet_tracks)), 2
        ):
            track_1 = formatted_poselet_tracks[track_index_1]
            track_2 = formatted_poselet_tracks[track_index_2]
            # Find common frames between two tracks
            merged_tracks = pd.merge(
                pd.DataFrame(track_1), pd.DataFrame(track_2), on=0
            )
            # If they have common frames add it to the list of combinations
            if merged_tracks.size:
                track_1 = np.array(merged_tracks["1_x"].tolist())
                track_2 = np.array(merged_tracks["1_y"].tolist())
                combined_tracks.append(
                    [
                        # Combinations of 2 of normalized tracked poselets
                        [track_1, track_2],
                        # Corresponding track indices
                        [track_index_1, track_index_2],
                        # Corresponding frame indices.
                        [merged_tracks[0].min(), merged_tracks[0].max()],
                    ]
                )

        return combined_tracks
