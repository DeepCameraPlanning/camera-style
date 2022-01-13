import os
import os.path as osp
from PIL import Image
from typing import Any, Callable, Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.utils.utils import load_csv


class BaseDataset(Dataset):
    """Load samples from a preprocessed dataset.

    :param dataset_dir: path to the dataset directory.
    :param set_mode: train, val or test set.
    :param n_frames: number of frames in a sample.
    :param transform: transformation to apply to frames.
    """

    def __init__(
        self,
        dataset_dir: str,
        n_frames: int,
        set_mode: str,
        transform: Callable,
    ):
        super().__init__()

        self._dataset_dir = dataset_dir

        self._n_frames = n_frames
        self._transform = transform

        self._frame_video_dir = osp.join(dataset_dir, "frame_clips", set_mode)
        self._shot_boundary_dir = osp.join(dataset_dir, "shot_boundaries")

        self._video_infos = self._get_video_infos()
        self._sample_infos = self._get_sample_infos()

    def _get_video_infos(self) -> List[Dict[str, Any]]:
        """Get frame paths for each video of the dataset.

        :return: a list of videos with:
            - `video_name`: name of the video.
            - `frame_paths`: paths of the video's frames.
            - `shot_boundaries`: last frame indices of each video's shots.
        """
        video_names = os.listdir(self._frame_video_dir)

        video_infos = []
        for video_name in sorted(video_names):
            frame_dir = osp.join(self._frame_video_dir, video_name)
            frame_paths = [
                osp.join(frame_dir, f) for f in sorted(os.listdir(frame_dir))
            ]

            shot_path = osp.join(self._shot_boundary_dir, video_name)
            shot_path += ".csv"
            shot_boundaries = load_csv(shot_path, header=None)[0].tolist()

            video_infos.append(
                {
                    "video_name": video_name,
                    "frame_paths": frame_paths,
                    "shot_boundaries": shot_boundaries,
                }
            )

        return video_infos

    def _get_sample_infos(self) -> List[Dict[str, Any]]:
        """TODO"""
        sample_infos = None
        return sample_infos

    @staticmethod
    def _load_frames(
        frame_paths: List[str],
        n_frames: int,
        transform: Callable,
    ) -> torch.Tensor:
        """
        Load frames and and apply transformations. Given their paths
        (C, T, W, H), and pads with zeros such that T == n_frames.
        """
        if frame_paths:
            frames = torch.stack(
                [
                    transform(Image.open(frame_path))
                    for frame_path in frame_paths
                ]
            )

        # Pad the tensor with zeros if the number of frames is too low
        n_current_frames = len(frames)
        if n_current_frames < n_frames:
            n_zeros = n_frames - n_current_frames
            frames = F.pad(frames.T, (n_zeros, 0), "constant", 0).T
        frames = frames.permute(1, 0, 2, 3)

        return frames

    def __len__(self) -> int:
        return len(self._sample_infos)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return the `index`-th sample.

        :return: a sample with:
            - `video_name`: name of the video containing the sample.
            - `shot_index`: index of the shot containing the sample.
            - `frames`: frames of the anchor.
        """
        sample_info = self._sample_infos[index]
        # Load anchor and positive frames
        frames = self._load_frames(
            sample_info["frame_paths"],
            self._n_frames,
            self._transform,
        )

        sample_data = {
            "video_name": sample_info["video_name"],
            "shot_index": sample_info["shot_index"],
            "frames": frames,
        }

        return sample_data
