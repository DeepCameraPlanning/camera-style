import os
import os.path as osp
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.file_utils import load_pth


class TripletFlowDataset(Dataset):
    """Load triplet samples from precomputed flows.

    :param clip_dirnames: list of clip directory names to load.
    :param unity_dir: path to the directory with precomputed Unity flows.
    :param prcpt_dir: path to the directory with precomputed flows.
    :param n_frames: number of flow frames in a sample.
    :param stride: number of flow frames between 2 consecutive samples.
    """

    def __init__(
        self,
        clip_dirnames: str,
        unity_dir: str,
        prcpt_dir: str,
        n_frames: int,
        stride: int,
    ):
        super().__init__()

        self._clip_dirnames = clip_dirnames
        self._unity_dir = unity_dir
        self._prcpt_dir = prcpt_dir

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
        """Get Unity and prcpt precomputed flow path for all samples.

        :return: a list of clips with:
            - `clip_name`: name of the clip.
            - `unity_paths`: paths of the Unity precomputed flows.
            - `prcpt_paths`: paths of the prcpt precomputed flows.
        """
        clip_infos = []
        for clip_dirname in sorted(self._clip_dirnames):
            unity_clip_dir = osp.join(self._unity_dir, clip_dirname)
            prcpt_clip_dir = osp.join(self._prcpt_dir, clip_dirname)

            unity_paths, prcpt_paths = [], []
            flow_names = os.listdir(unity_clip_dir)
            for flow_filename in sorted(flow_names):
                unity_paths.append(osp.join(unity_clip_dir, flow_filename))
                prcpt_paths.append(osp.join(prcpt_clip_dir, flow_filename))

            clip_infos.append(
                {
                    "clip_name": clip_dirname,
                    "unity_paths": unity_paths,
                    "prcpt_paths": prcpt_paths,
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
            prcpt_paths = clip_info["prcpt_paths"]

            unity_gen = self._split_chunks(
                unity_paths, self._stride, self._n_frames
            )
            prcpt_gen = self._split_chunks(
                prcpt_paths, self._stride, self._n_frames
            )
            for chunk_index, chunks in enumerate(zip(unity_gen, prcpt_gen)):
                unity_chunk, prcpt_chunk = chunks
                if len(unity_chunk) != self._n_frames:
                    break
                frame_start = self._stride * chunk_index
                frame_end = frame_start + self._n_frames - 1
                chunk_infos = np.array(
                    {
                        "clip_name": clip_name
                        + f"/{frame_start:04}-{frame_end:04}",
                        "unity_chunk_paths": unity_chunk,
                        "prcpt_chunk_paths": prcpt_chunk,
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
            positive_paths = positive_sample["prcpt_chunk_paths"]
            # Select another random sample from a different clip
            clip_mask = sample_clipnames != clip_name
            negative_sample = np.random.choice(sample_splits[clip_mask], 1)[0]
            negative_clipname = negative_sample["clip_name"]
            negative_paths = negative_sample["prcpt_chunk_paths"]

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

        anc_paths = sample_info["anchor_paths"]
        pos_paths = sample_info["positive_paths"]
        neg_paths = sample_info["negative_paths"]

        anc_flows = self._load_flows(anc_paths)
        pos_flows = self._load_flows(pos_paths)
        neg_flows = self._load_flows(neg_paths)

        sample_data = {
            "positive_clipname": sample_info["positive_clipname"],
            "negative_clipname": sample_info["negative_clipname"],
            "anchor_flows": anc_flows,
            "positive_flows": pos_flows,
            "negative_flows": neg_flows,
        }

        return sample_data
