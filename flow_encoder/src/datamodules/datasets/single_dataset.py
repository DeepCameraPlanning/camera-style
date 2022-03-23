import os
import os.path as osp
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.file_utils import load_pth


class SingleFlowDataset(Dataset):
    """Load a single sample from precomputed flows.

    :param flow_dir: path to the directory with precomputed flows.
    :param n_frames: number of flow frames in a sample.
    :param stride: number of flow frames between 2 consecutive samples.
    """

    def __init__(
        self,
        flow_dir: str,
        n_frames: int,
        stride: int,
    ):
        super().__init__()

        self._flow_dir = flow_dir
        self._clip_dirnames = sorted(os.listdir(flow_dir))

        self._n_frames = n_frames
        self._stride = stride

        self._clip_infos = self._get_clip_infos()
        self._sample_infos = self._get_sample_infos()

    @staticmethod
    def _split_chunks(array: List[Any], stride: int, chunk_size: int):
        """Yield successive n-sized chunks from `array`."""
        for i in range(0, len(array), stride):
            yield array[i : i + chunk_size]

    @staticmethod
    def _load_flows(flow_paths: List[str]) -> torch.Tensor:
        """Load flows. Output shape: (C, T, W, H)."""
        flows = torch.stack([load_pth(flow_path) for flow_path in flow_paths])
        flows = flows.permute([3, 0, 1, 2])

        return flows

    def _get_clip_infos(self) -> List[Dict[str, Any]]:
        """Get precomputed flow paths for all samples.

        :return: a list of clips with:
            - `clip_name`: name of the clip.
            - `flow_paths`: paths of the flow precomputed flows.
        """
        clip_infos = []
        for clip_dirname in sorted(self._clip_dirnames):
            flow_clip_dir = osp.join(self._flow_dir, clip_dirname)

            flow_paths = []
            flow_names = os.listdir(flow_clip_dir)
            for flow_filename in sorted(flow_names):
                flow_paths.append(osp.join(flow_clip_dir, flow_filename))

            clip_infos.append(
                {
                    "clip_name": clip_dirname,
                    "flow_paths": flow_paths,
                }
            )

        return clip_infos

    def _get_sample_infos(self) -> List[Dict[str, Any]]:
        """
        Each sample is composed of `n_frames` consecutive flows.

        :return: a list of samples with:
            - `clipname`: sample clip name.
            - `flow_paths`: paths of the `n_frames` flows.
        """
        # Start by splitting each clip into chunks of `n_frames`
        sample_splits, sample_clipnames = np.empty((0,)), np.empty((0,))
        for clip_info in self._clip_infos:
            clip_name = clip_info["clip_name"]
            flow_paths = clip_info["flow_paths"]

            flow_gen = self._split_chunks(
                flow_paths, self._stride, self._n_frames
            )
            for chunk_index, flow_chunk in enumerate(flow_gen):
                if len(flow_chunk) != self._n_frames:
                    break
                frame_start = self._stride * chunk_index
                frame_end = frame_start + self._n_frames - 1
                chunk_infos = np.array(
                    {
                        "clip_name": clip_name
                        + f"/{frame_start:04}-{frame_end:04}",
                        "flow_chunk_paths": flow_chunk,
                    }
                ).reshape(1)
                sample_splits = np.hstack([sample_splits, chunk_infos])
                sample_clipnames = np.hstack([sample_clipnames, clip_name])

        # Then generate the triplet samples (anchor. positive, negative)
        sample_infos = []
        for sample in sample_splits:
            sample_infos.append(
                {
                    "clipname": sample["clip_name"],
                    "flow_paths": sample["flow_chunk_paths"],
                }
            )

        return sample_infos

    def __len__(self) -> int:
        return len(self._sample_infos)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return the `index`-th sample.

        :return: a sample with:
            - `clipname`: positive and anchor sample clip name.
            - `flows`: flows of the anchor.
        """
        sample_info = self._sample_infos[index]

        flow_paths = sample_info["flow_paths"]
        flows = self._load_flows(flow_paths)
        sample_data = {"clipname": sample_info["clipname"], "flows": flows}

        return sample_data
