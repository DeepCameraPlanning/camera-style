import os
import os.path as osp
from typing import Any, Callable, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.utils import load_pth


class TripletFlowDataset(Dataset):
    """Load triplet samples from a precomputed flows.

    :param unity_dir: path to the directory with precomputed Unity flows.
    :param raft_dir: path to the directory with precomputed RAFT flows.
    :param n_frames: number of frames in a sample.
    :param transform: transformation to apply to frames.
    """

    def __init__(
        self,
        unity_dir: str,
        raft_dir: str,
        n_frames: int,
        transform: Callable,
    ):
        super().__init__()

        self._unity_dir = unity_dir
        self._raft_dir = raft_dir

        self._n_frames = n_frames
        self._transform = transform

        self._clip_infos = self._get_clip_infos()
        self._sample_infos = self._get_sample_infos()

    @staticmethod
    def _split_chunks(array: List[Any], chunk_size: int):
        """Yield successive n-sized chunks from `array`."""
        for i in range(0, len(array), chunk_size):
            yield array[i : i + chunk_size]

    @staticmethod
    def _load_flows(
        flow_paths: List[str],
        transform: Callable,
    ) -> torch.Tensor:
        """Load flows and and apply transformations, shape: (C, T, W, H)."""
        flows = torch.stack(
            [load_pth(flow_path) for flow_path in flow_paths]
        ).permute([3, 0, 1, 2])

        if transform is not None:
            flows = [transform(flow) for flow in flows]

        return flows

    def _get_clip_infos(self) -> List[Dict[str, Any]]:
        """Get Unity and RAFT precomputed flow path for all samples.

        :return: a list of clips with:
            - `clip_name`: name of the clip.
            - `unity_paths`: paths of the Unity precomputed flows.
            - `raft_paths`: paths of the RAFT precomputed flows.
        """
        clip_infos = []
        clip_names = os.listdir(self._unity_dir)
        for clip_dirname in sorted(clip_names):
            unity_clip_dir = osp.join(self._unity_dir, clip_dirname)
            raft_clip_dir = osp.join(self._raft_dir, clip_dirname)

            unity_paths, raft_paths = [], []
            flow_names = os.listdir(unity_clip_dir)
            for flow_filename in sorted(flow_names):
                unity_paths.append(osp.join(unity_clip_dir, flow_filename))
                raft_paths.append(osp.join(raft_clip_dir, flow_filename))

            clip_infos.append(
                {
                    "clip_name": clip_dirname,
                    "unity_paths": unity_paths,
                    "raft_paths": raft_paths,
                }
            )

        return clip_infos

    def _get_sample_infos(self) -> List[Dict[str, Any]]:
        """
        Generate a list of triplet samples (anchor, positive, negative). Each
        sample is composed of `n_frames` consecutive flows. Negative samples
        are randomly selected among samples of another clip.

        :return: a list of samples with:
            - `positive_clipname`: positive and anchor sample clip name.
            - `negative_clipname`: negative sample clip name.
            - `anchor_paths`: paths of the `n_frames` anchor flows.
            - `positive_paths`: paths of the `n_frames` positive flows.
            - `negative_paths`: paths of the `n_frames` negative flows.
        """
        # Start by splitting each clip into chunks of `n_frames`
        sample_splits, sample_clipnames = np.empty((0,)), np.empty((0,))
        for clip_info in self._clip_infos:
            clip_name = clip_info["clip_name"]
            unity_paths = clip_info["unity_paths"]
            raft_paths = clip_info["raft_paths"]

            unity_gen = self._split_chunks(unity_paths, self._n_frames)
            raft_gen = self._split_chunks(raft_paths, self._n_frames)
            for unity_chunk, raft_chunk in zip(unity_gen, raft_gen):
                if len(unity_chunk) != self._n_frames:
                    break
                chunk_infos = np.array(
                    {
                        "clip_name": clip_name,
                        "unity_chunk_paths": unity_chunk,
                        "raft_chunk_paths": raft_chunk,
                    }
                ).reshape(1)
                sample_splits = np.hstack([sample_splits, chunk_infos])
                sample_clipnames = np.hstack([sample_clipnames, clip_name])

        # Then generate the triplet samples (anchor. positive, negative)
        sample_infos = []
        for positive_sample in sample_splits:
            # Get anchor and positive infos from the current sample
            positive_clipname = positive_sample["clip_name"]
            anchor_paths = positive_sample["unity_chunk_paths"]
            positive_paths = positive_sample["raft_chunk_paths"]
            # Select another random sample from a different clip
            clip_mask = sample_clipnames != clip_name
            negative_sample = np.random.choice(sample_splits[clip_mask], 1)[0]
            negative_clipname = negative_sample["clip_name"]
            negative_paths = negative_sample["raft_chunk_paths"]

            sample_infos.append(
                {
                    "positive_clipname": positive_clipname,
                    "negative_clipname": negative_clipname,
                    "anchor_paths": anchor_paths,
                    "positive_paths": positive_paths,
                    "negative_paths": negative_paths,
                }
            )

        return sample_infos

    def __len__(self) -> int:
        return len(self._sample_infos)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return the `index`-th sample.

        :return: a sample with:
            - `positive_clipname`: positive and anchor sample clip name.
            - `negative_clipname`: negative sample clip name.
            - `anchor_flows`: flows of the anchor.
            - `positive_flows`: flows of the positive.
            - `negative_flows`: flows of the negative.
        """
        sample_info = self._sample_infos[index]

        anchor_paths = sample_info["anchor_paths"]
        positive_paths = sample_info["positive_paths"]
        negative_paths = sample_info["negative_paths"]

        anchor_flows = self._load_flows(anchor_paths, self._transform)
        positive_flows = self._load_flows(positive_paths, self._transform)
        negative_flows = self._load_flows(negative_paths, self._transform)

        sample_data = {
            "positive_clipname": sample_info["positive_clipname"],
            "negative_clipname": sample_info["negative_clipname"],
            "anchor_flows": anchor_flows,
            "positive_flows": positive_flows,
            "negative_flows": negative_flows,
        }

        return sample_data
