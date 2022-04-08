from collections import defaultdict
import os
import os.path as osp
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.file_utils import load_pth, load_pickle
from utils.feature_utils import compute_bbox_area


class LatentBboxDataset(Dataset):
    """Load pre-computed flow features, flow frames, and bounding-boxes.

    :param clip_dirnames: list of clip directory names to load.
    :param bbox_dir: directory containing pre-extracted detections.
    :param flow_dir: directory containing pre-extracted flows.
    :param frame_dir: directory containing raw_frames.
    """

    def __init__(
        self,
        clip_dirnames: str,
        bbox_dir: str,
        flow_dir: str,
        frame_dir: str,
        stride: int,
        n_frames: int,
    ):
        super().__init__()

        self._clip_dirnames = clip_dirnames
        self._bbox_dir = bbox_dir
        self._flow_dir = flow_dir
        self._frame_dir = frame_dir

        self._stride = stride
        self._n_frames = n_frames

        self._clip_infos = self._get_clip_infos()
        self._sample_infos, self._sample_keys = self._get_sample_infos()

    @staticmethod
    def _load_flows(flow_paths: List[str]) -> torch.Tensor:
        """Load flows. Output shape: (C, T, W, H)."""
        flows = torch.stack([load_pth(flow_path) for flow_path in flow_paths])
        flows = flows.permute([3, 0, 1, 2])

        return flows

    @staticmethod
    def _load_bboxes(bbox_path: str) -> Dict[int, List[np.array]]:
        """Load bboxes and map each frame to its bboxes."""
        # Load bboxes: track -> bboxes
        (height, width), raw_bboxes = load_pickle(bbox_path)

        processed_bboxes = defaultdict(list)
        for track in raw_bboxes:
            for bbox in track:
                frame_index = bbox[0]
                bbox_coordinates = bbox[1:]
                norm_bbox = np.zeros_like(bbox_coordinates)
                norm_bbox[[0, 2]] = bbox_coordinates[[0, 2]] / width
                norm_bbox[[1, 3]] = bbox_coordinates[[1, 3]] / height
                norm_bbox = np.where(
                    bbox_coordinates < 0,
                    np.zeros_like(norm_bbox),
                    norm_bbox,
                )
                processed_bboxes[frame_index].append(norm_bbox)

        return processed_bboxes

    @staticmethod
    def _load_character_mask(
        flow: torch.Tensor, bboxes: List[np.array]
    ) -> torch.Tensor:
        """Load a flow frame and mask what's outside of bboxes."""
        height, width, _ = flow.shape
        character_mask = torch.zeros_like(flow)
        for bbox in bboxes:
            regular_bbox = np.zeros_like(bbox)
            regular_bbox[[0, 2]] = bbox[[0, 2]] * width
            regular_bbox[[1, 3]] = bbox[[1, 3]] * height
            x1, y1, x2, y2 = regular_bbox.astype(int)
            character_mask[y1:y2, x1:x2] = flow[y1:y2, x1:x2]

        return character_mask

    @staticmethod
    def _get_largest_bbox(bboxes: List[np.array]) -> torch.Tensor:
        """Select the largest bbox or if empty, return a null bbox."""
        if len(bboxes) == 0:
            return torch.zeros(4, dtype=torch.float64)

        bbox_areas = compute_bbox_area(np.array(bboxes))
        largest_bbox = torch.from_numpy(bboxes[np.argmax(bbox_areas)])
        return largest_bbox

    @staticmethod
    def _get_paths(root_dir: str) -> List[str]:
        """Get the list of filenames in a directory."""
        filenames = os.listdir(root_dir)
        paths = [osp.join(root_dir, name) for name in sorted(filenames)]
        return paths

    @staticmethod
    def _split_chunks(array: List[Any], stride: int, chunk_size: int):
        """Yield successive n-sized chunks from `array`."""
        for i in range(0, len(array), stride):
            yield array[i : i + chunk_size]

    def _get_clip_infos(self) -> List[Dict[str, Any]]:
        """Get feature, bbox and flow paths for each clip.

        :return: a list of clips with:
            - `clip_name`: name of the clip.
            - `bbox_path`: path to the precomputed bboxes.
            - `flow_paths`: paths to the precomputed flows.
            - `frame_paths`: paths to the raw_frames.
        """
        clip_infos = []
        for clip_dirname in sorted(self._clip_dirnames):
            bbox_path = osp.join(self._bbox_dir, clip_dirname + ".pk")

            flow_dir = osp.join(self._flow_dir, clip_dirname)
            flow_paths = self._get_paths(flow_dir)

            frame_dir = osp.join(self._frame_dir, clip_dirname)
            frame_paths = self._get_paths(frame_dir)

            clip_infos.append(
                {
                    "clip_name": clip_dirname,
                    "bbox_path": bbox_path,
                    "flow_paths": flow_paths,
                    "frame_paths": frame_paths,
                }
            )

        return clip_infos

    def _get_sample_infos(
        self,
    ) -> Tuple[Dict[Tuple[str, int], Dict[str, Any]], List[str]]:
        """
        TODO...

        :return: a map from (`clip_name`, `chunk_index`) to:
            - `keyframe_index`: index of the keyframe in the chunk.
            - `key_bboxes`:
            - `key_flow_path`:
            - `key_frame_path`:
            - `chunk_flow_paths`:
            And return also the list of all (`clip_name`, `chunk_index`).
        """
        sample_infos = {}
        for clip_info in self._clip_infos:
            clip_name = clip_info["clip_name"]
            bbox_path = clip_info["bbox_path"]
            flow_paths = clip_info["flow_paths"]
            frame_paths = clip_info["frame_paths"]

            clip_bboxes = self._load_bboxes(bbox_path)

            flow_gen = self._split_chunks(
                sorted(flow_paths), self._stride, self._n_frames
            )
            for chunk_index, chunk_flow_paths in enumerate(flow_gen):
                if len(chunk_flow_paths) != self._n_frames:
                    break
                frame_start = self._stride * chunk_index
                frame_end = frame_start + self._n_frames - 1
                # Get chunk path and bboxes for a keyframe
                keyframe_index = frame_start + ((frame_end - frame_start) // 2)
                key_flow_path = sorted(flow_paths)[keyframe_index]
                key_frame_path = sorted(frame_paths)[keyframe_index]
                key_bboxes = clip_bboxes[keyframe_index]
                if len(key_bboxes) > 0:
                    # Store chunk information
                    sample_infos[(clip_name, chunk_index)] = {
                        "keyframe_index": keyframe_index,
                        "key_bboxes": key_bboxes,
                        "key_flow_path": key_flow_path,
                        "key_frame_path": key_frame_path,
                        "chunk_flow_paths": chunk_flow_paths,
                    }

        # Get consecutive samples with bboxes
        sample_keys = []
        for clip_name, chunk_index in sample_infos.keys():
            if (clip_name, chunk_index + 1) in sample_infos.keys():
                sample_keys.append(
                    [(clip_name, chunk_index), (clip_name, chunk_index + 1)]
                )

        return sample_infos, sample_keys

    def __len__(self) -> int:
        return len(self._sample_keys)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return the `index`-th sample.

        :return: a sample with:
            - `clip_name`: name of clip.
            - `keyframe_indices`: indices of input and target chunk keyframes.
            - `input_feature`: flow feature of the input chunk.
            - `input_character_mask`: char mask of the input chunk keyframe.
            - `target_bbox`: bbox coordinates of the target chunk keyframe.
        """
        input_key, target_key = self._sample_keys[index]
        input_clip_name, chunk_index = input_key

        # Load input data
        input_sample_infos = self._sample_infos[input_key]
        input_keyframe = input_sample_infos["keyframe_index"]
        input_flow = load_pth(input_sample_infos["key_flow_path"])
        input_bboxes = input_sample_infos["key_bboxes"]
        input_character_mask = self._load_character_mask(
            input_flow, [self._get_largest_bbox(input_bboxes)]
        )
        input_frame = cv2.resize(
            cv2.imread(input_sample_infos["key_frame_path"]), (224, 224)
        )
        input_flows = self._load_flows(input_sample_infos["chunk_flow_paths"])

        # Load target data
        target_sample_infos = self._sample_infos[target_key]
        target_keyframe = target_sample_infos["keyframe_index"]
        target_bboxes = self._get_largest_bbox(
            target_sample_infos["key_bboxes"]
        )
        target_frame = cv2.resize(
            cv2.imread(target_sample_infos["key_frame_path"]), (224, 224)
        )

        sample_data = {
            "clip_name": input_clip_name,
            "keyframe_indices": (input_keyframe, target_keyframe),
            "input_flows": input_flows,
            "input_character_mask": input_character_mask,
            "target_bbox": target_bboxes,
            "input_frame": input_frame,
            "target_frame": target_frame,
        }

        return sample_data
