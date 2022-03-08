from itertools import product
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
from detectron2.modeling import build_model
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
import torch
from torch.utils.data import DataLoader

from lib.sort.sort import Sort
from src.utils.diverse_utils import compute_bbox_iou, get_patches


class SegmentationModel(LightningModule):
    def __init__(self, config):
        super(SegmentationModel, self).__init__()
        model = build_model(config)
        DetectionCheckpointer(model).load(config.MODEL.WEIGHTS)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        detection = self.model(x)
        return detection


class PeopleDetector:
    """
    Detect humans within a given frame.
    Code adapted from: https://github.com/facebookresearch/detectron2.

    :param score_threshold: threshold to consider a detection.
    :param batch_size: how many samples per batch to load.
    :param num_gpus: how many gpus to use for inference.
    :param num_nodes: how many nodes to use for inference.
    :param num_workers: how many subprocesses to use for data loading.
    :param verbose: whether to display a progress bar or not.
    """

    def __init__(
        self,
        score_threshold: float = 0.9,
        batch_size: int = 10,
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

        config_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        self._config = get_cfg()
        self._config.merge_from_file(model_zoo.get_config_file(config_file))
        # Set threshold for this model
        self._config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
        # Get pre-trained weights
        self._config.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)

        model = SegmentationModel(self._config).eval()
        self.model = model
        self.trainer = Trainer(
            gpus=num_gpus,
            num_nodes=num_nodes,
            strategy="dp",
            logger=False,
            enable_progress_bar=enable_progress_bar,
        )

        # Initialize a bounding-box tracker
        self.tracker = SortTracker()

    def _preprocess_frames(self, frames: List[np.array]) -> List[torch.Tensor]:
        """Pre-process a list of frames."""
        processer = T.ResizeShortestEdge(
            [
                self._config.INPUT.MIN_SIZE_TEST,
                self._config.INPUT.MIN_SIZE_TEST,
            ],
            self._config.INPUT.MAX_SIZE_TEST,
        )
        processed_frames = []
        for frame in frames:
            height, width = frame.shape[:2]
            image = processer.get_transform(frame).apply_image(frame)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            processed_frames.append(
                {"image": image, "height": height, "width": width}
            )

        return processed_frames

    @staticmethod
    def _postprocess_maps(detections: torch.Tensor) -> List[np.array]:
        """Post-process a list of detections."""
        bboxes, masks, scores = [], [], []
        for batch_detections in detections:
            for frame_detection in batch_detections:
                # Select only human instances
                instances = frame_detection["instances"][
                    frame_detection["instances"].pred_classes == 0
                ]

                # Get bboxes and masks and convert it to numpy arrays
                bboxes.append(instances.pred_boxes.tensor.cpu().numpy())
                masks.append(instances.pred_masks.cpu().numpy())
                scores.append(instances.scores.cpu().numpy())

        return bboxes, masks, scores

    @staticmethod
    def _collate_batch(batch):
        test_batch = [b for b in batch]
        return test_batch

    def detect_people(
        self, frames: List[np.array]
    ) -> Tuple[np.array, np.array, np.array]:
        """
        Detect all human instances on given frames.
        Code adapted from: https://github.com/facebookresearch/detectron2.

        :param frames: list of BGR frames.
        :return: bounding-boxes coordinates, masks and confidence scores.
        """
        preprocessed_frames = self._preprocess_frames(frames)
        frame_loader = DataLoader(
            preprocessed_frames,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate_batch,
        )
        detections = self.trainer.predict(self.model, frame_loader)
        bboxes, masks, scores = self._postprocess_maps(detections)

        return bboxes, masks, scores

    def get_bboxtracks(
        self, bboxes: List[np.array], scores: List[np.array]
    ) -> List[np.array]:
        """Get bounding-box tracks."""
        bbox_tracks, index_tracks = self.tracker.track_bboxes(bboxes, scores)
        # Store `index_tracks` to reuse it for mask tracking
        self._index_tracks = index_tracks

        return bbox_tracks

    def get_masktracks(self, masks: List[np.array]) -> List[np.array]:
        """Get mask tracks (`get_bboxtracks` should be run before)."""
        mask_tracks = self.tracker.track_bboxes(masks, self._index_tracks)

        return mask_tracks

    def get_peoplefeatures(
        self,
        frames: List[np.array],
        bboxes: List[np.array],
        scores: List[np.array],
        featuremap_dims: Tuple[int, int],
    ) -> List[np.array]:
        """
        Compute bounding-box features: create a binary grid with 1s where
        bounding-boxes are located.

        :param frames: list of frames.
        :param bboxes: list of detected bounding-boxes.
        :param scores: list of associated confidence scores.
        :param featuremap_dims: dimensions of the output layer grid.
        :return: binary grids with bounding-boxes.
        """
        n_row, n_col = featuremap_dims
        bbox_features = []
        for frame_index, frame in enumerate(frames):
            bbox_patches = get_patches(
                np.zeros(frame.shape[:2]), (n_row, n_col)
            )
            y_stride, x_stride = bbox_patches.shape[2:]

            frame_bboxes = bboxes[frame_index]
            bbox_map = np.zeros((n_row, n_col))
            for bbox in frame_bboxes:
                x1, y1, x2, y2 = bbox
                # ROI pooling
                pooled_x1, pooled_x2 = np.array([x1, x2]) // x_stride
                pooled_y1, pooled_y2 = np.array([y1, y2]) // y_stride

                pooled_indices = np.array(
                    [
                        (y, x)
                        for x, y in product(
                            range(int(pooled_x1), int(pooled_x2)),
                            range(int(pooled_y1), int(pooled_y2)),
                        )
                    ]
                )
                bbox_pooled_indices = np.moveaxis(pooled_indices, -1, 0)
                bbox_map[tuple(bbox_pooled_indices)] = 1

            bbox_features.append(bbox_map)

        return bbox_features


class Tracker:
    def __init__(
        self,
        iou_matching_threshold: float = 0.5,
    ):
        """
        :param iou_matching_threshold: minimum overlap between bounding-boxes.
        """
        self.iou_matching_threshold = iou_matching_threshold

    @staticmethod
    def detect_inout(bbox_track: np.array, n_frames: int) -> Tuple[int, int]:
        """
        Check wether the tracked bounding-box enter after the shot starts or
        exit before the shot ends.
        """
        start_frame = bbox_track[0, 0]
        end_frame = bbox_track[-1, 0]

        in_frame = start_frame if start_frame > 0 else None
        out_frame = end_frame if end_frame < (n_frames - 1) else None

        return in_frame, out_frame

    def match_bbox_annotation(
        self,
        annotated_bbox: np.array,
        bbox_tracks: List[np.array],
        frame_index: int,
    ) -> Tuple[int, np.array]:
        """Given a ground-truth bounding-box, find the match track.

        :param annotated_bbox: ground-truth bounding-box.
        :param bbox_tracks: list of tracks with frame index and bounding-box.
        :param frame_index: index of the frame with the annotated bounding-box.
        :return: if match, list of tracked bounding-boxes.
        """
        ious = []
        for track in bbox_tracks:
            # Retrieve the tracked bbox of the annotated frame
            (box_index,) = np.where(track[:, 0] == frame_index)

            # Check if there is a tracked bbox for this frame
            if track[box_index].size > 0:
                # Compute IoU between ground-tuth and tracked bboxes
                iou = compute_bbox_iou(
                    track[box_index][:, -4:], annotated_bbox
                )
                ious.append(iou[0])
            else:
                ious.append(-1)

        # Get the highest IoU
        best_index = np.argmax(ious)
        if ious[best_index] > self.iou_matching_threshold:
            return best_index, bbox_tracks[best_index]

        return -1, np.array([])

    @staticmethod
    def interpolate_bboxes(
        track: np.array, first_frame: int, last_frame: int
    ) -> np.array:
        """Interpolate missing bounding-boxes."""
        active_track = [[k] for k in range(first_frame, last_frame)]
        for bbox_index in range(len(track)):
            frame_index = int(track[bbox_index][0])
            bbox = track[bbox_index][1:]
            # Update active track
            active_track[frame_index - first_frame] += bbox.tolist()

            # Check if there is a gap between the last and current frame
            if (bbox_index > 0) and (
                active_track[bbox_index - 1][0] != frame_index - 1
            ):
                # Interpolate missing frame
                last_index = int(track[bbox_index - 1][0])
                last_bbox = track[bbox_index - 1][1:]
                for missing_index in range(last_index + 1, frame_index):
                    alpha = (frame_index - missing_index) / (
                        frame_index - last_index
                    )
                    active_track[missing_index - first_frame] += (
                        alpha * last_bbox + (1 - alpha) * bbox
                    ).tolist()

        return np.array(active_track)

    @staticmethod
    def interpolate_masks(
        track: np.array, first_frame: int, last_frame: int
    ) -> np.array:
        """Interpolate missing bounding-boxes."""
        active_track = [[k] for k in range(first_frame, last_frame)]
        for mask_index in range(len(track)):
            frame_index = int(track[mask_index][0])
            mask = track[mask_index][1]
            # Update active track
            active_track[frame_index - first_frame] += [mask]

            # Check if there is a gap between the last and current frame
            if (mask_index > 0) and (
                active_track[mask_index - 1][0] != frame_index - 1
            ):
                # Interpolate missing frame
                last_index = int(track[mask_index - 1][0])
                last_mask = track[mask_index - 1][1]
                for missing_index in range(last_index + 1, frame_index):
                    active_track[missing_index - first_frame] += [
                        np.logical_and(last_mask, mask)
                    ]

        return np.array(active_track, dtype=object)


class NaiveTracker(Tracker):
    def __init__(
        self,
        iou_threshold: float = 0.5,
        max_frame_threshold: int = 10,
        min_frame_threshold: int = 10,
    ):
        """
        :param iou_threshold: minimum overlap between consecutive frames.
        :param max_frame_threshold: maximum number of skipped frame before
            considering a track over.
        :param min_frame_threshold: minimum number of frame to consider a
            track as relevant.
        """
        super().__init__()
        self.iou_threshold = iou_threshold
        self.max_frame_threshold = max_frame_threshold
        self.min_frame_threshold = min_frame_threshold

    @staticmethod
    def _merge_tracks(
        forward_tracks: List[List[np.array]],
        backward_tracks: List[List[np.array]],
        common_threshold: int = 5,
    ) -> List[np.array]:
        """
        Merge forward and backward tracks:
            - Check if the forward and backward tracks have common bboxes.
            - If they do, gather bboxes.
            - If duplicates for same frame index, keep highest score.

        :param forward_tracks: tracks obtained with a forward tracking.
        :param backward_tracks: tracks obtained with a backward tracking.
        :return: gathered tracks.
        """
        column_names = ["frame_index", "score", "x1", "y1", "x2", "y2"]
        merged_tracks = []
        f_visited, b_visited = [], []
        for f_track_index, f_track in enumerate(forward_tracks):
            f_track_df = pd.DataFrame(f_track, columns=column_names)

            for b_track_index, b_track in enumerate(backward_tracks):
                # Check whether current tracks have not been met already
                if not (f_track_index in f_visited) or not (
                    b_track_index in b_visited
                ):
                    b_track_df = pd.DataFrame(b_track, columns=column_names)
                    # Check whether tracks have common bboxes (inner join)
                    n_common_bboxes = pd.merge(f_track_df, b_track_df).shape[0]
                    if n_common_bboxes >= common_threshold:
                        f_visited.append(f_track_index)
                        b_visited.append(b_track_index)
                        # Merge tracks (outer join)
                        merged = pd.merge(f_track_df, b_track_df, how="outer")
                        # Remove duplicates for same frame index
                        merged_tracks.append(
                            merged[
                                merged.groupby("frame_index")[
                                    "score"
                                ].transform(max)
                                == merged["score"]
                            ]
                            .sort_values("frame_index")[
                                ["frame_index", "x1", "y1", "x2", "y2"]
                            ]
                            .to_numpy()
                        )

        return merged_tracks

    def _get_tracks(
        self,
        bboxes: List[np.array],
        scores: List[np.array],
        first_frame: int,
        last_frame: int,
    ) -> List[List[np.array]]:
        """
        Track bounding-boxes from a list. Forward and backward pass available.
        Code adapted from: https://github.com/AVAuco/laeonetplus/.

        :param bboxes: list of bounding-boxes to track.
        :param scores: list of segmentation scores for each bounding-box.
        :param first_frame: index of the first frame (backward: highest index).
        :param last_frame: index of the last frame (backward: lowest index)..
        :return: list of tracks with frame index and bounding-box.
        """
        # Whether to apply a forward or backward pass
        forward = True if (first_frame < last_frame) else False

        # Initialize first bounding-boxes to track
        pending_tracks = [
            [np.array([first_frame, score] + bbox.tolist())]
            for bbox, score in zip(bboxes[first_frame], scores[first_frame])
        ]

        completed_tracks = []
        # Whether to pick the last or the first element
        pick = -1 if forward else 0
        # Whether to iterate forward or backward
        direction = 1 if forward else -1
        for frame_index in range(
            first_frame + direction, last_frame + direction, direction
        ):
            # New bbox to track
            current_bboxes = bboxes[frame_index]
            current_scores = scores[frame_index]

            completed_track_indices = []
            # Whether to insert tracked bbox at the end or the begining
            order = frame_index if forward else 0
            for track_index, reference_track in enumerate(pending_tracks):
                # Compute IoU between the last tracked bbox and untracked ones
                ref_frame_index = reference_track[pick][0]
                reference_box = reference_track[pick][2:]
                ious = compute_bbox_iou(current_bboxes, reference_box)
                (possible_match,) = np.where(ious >= self.iou_threshold)

                # Check whether any bbox could match with the reference one
                if possible_match.size > 0:
                    # Pick the highest IoU and add it to the reference track
                    best_bbox_index = np.argmax(ious)
                    best_bbox = current_bboxes[best_bbox_index]
                    best_score = current_scores[best_bbox_index]
                    pending_tracks[track_index].insert(
                        order,
                        np.array(
                            [frame_index, best_score] + best_bbox.tolist()
                        ),
                    )
                    current_bboxes = np.delete(
                        current_bboxes, best_bbox_index, axis=0
                    )
                    current_scores = np.delete(
                        current_scores, best_bbox_index, axis=0
                    )

                # No matching bbox, check if the reference track is over
                else:
                    if (
                        abs(frame_index - ref_frame_index)
                        >= self.max_frame_threshold
                    ):
                        completed_track_indices.append(track_index)

            # Add to completed tracks those that are over. Iterate reversly
            # to avoid indexing issues
            for track_index in completed_track_indices[::-1]:
                completed_tracks.append(pending_tracks.pop(track_index))

            # Create new tracks for residual unmatched bboxes
            for bbox, score in zip(current_bboxes, current_scores):
                pending_tracks.append(
                    [np.array([frame_index, score] + bbox.tolist())],
                )

        # Add residual tracks to completed ones
        completed_tracks += pending_tracks

        return completed_tracks

    def _process_tracks(
        self,
        bbox_tracks: List[List[Tuple[int, np.array]]],
    ) -> List[np.array]:
        """
        Process tracks:
            - Remove short tracks.
            - Interpolate missing frames.
            - Remove duplicate tracks.
        Code adapted from: https://github.com/AVAuco/laeonetplus/.

        :param bbox_tracks: list of tracks with frame index and bounding-box.
        :return: processed list of tracks with frame index and bounding-box.
        """

        def is_duplicate(
            processed_tracks: List[np.array], active_track: np.array
        ) -> bool:
            """Check if active track is already in processed tracks."""
            for track in processed_tracks:
                if np.array_equal(track, active_track):
                    return True
            return False

        processed_tracks = []
        for track_index, track in enumerate(bbox_tracks):
            first_frame, last_frame = int(track[0][0]), int(track[-1][0]) + 1
            track_length = last_frame - first_frame
            # Remove track if it is too short
            if track_length <= self.min_frame_threshold:
                continue

            # Interpolate missing bounding-boxes
            active_track = self.interpolate_bboxes(
                track, first_frame, last_frame
            )

            # Remove duplicate tracks
            if is_duplicate(processed_tracks, active_track):
                continue

            processed_tracks.append(active_track)

        return processed_tracks

    def track_bboxes(
        self, bboxes: List[np.array], scores: List[np.array]
    ) -> List[np.array]:
        """
        Track bounding-boxes according to their detection scores.
        Naive implementation with a forward, backward pass and IoU.
        """
        n_frames = len(bboxes)
        frame_threshold = max(int(0.05 * n_frames), 10)

        forward_tracks = self._get_tracks(
            bboxes,
            scores,
            first_frame=0,
            last_frame=n_frames - 1,
            frame_threshold=frame_threshold,
        )
        backward_tracks = self._get_tracks(
            bboxes,
            scores,
            first_frame=n_frames - 1,
            last_frame=0,
            frame_threshold=frame_threshold,
        )
        raw_tracks = self._merge_tracks(forward_tracks, backward_tracks)
        tracks = self._process_tracks(raw_tracks)

        return tracks


class SortTracker(Tracker):
    def __init__(
        self,
        iou_threshold: float = 0.5,
        max_frame_threshold: int = 10,
        min_frame_threshold: int = 10,
    ):
        """
        :param iou_threshold: minimum overlap between consecutive frames.
        :param max_frame_threshold: maximum number of skipped frame before
            considering a track over.
        :param min_frame_threshold: minimum number of frame to consider a
            track as relevant.
        """
        super().__init__()
        self.iou_threshold = iou_threshold
        self.max_frame_threshold = max_frame_threshold
        self.min_frame_threshold = min_frame_threshold

    def track_bboxes(
        self, bboxes: List[np.array], scores: List[np.array]
    ) -> List[np.array]:
        """
        Track bounding-boxes according to their detection scores.
        Code adapted from: https://github.com/abewley/sort.
        """
        tracker = Sort(
            max_age=self.max_frame_threshold,
            min_hits=self.min_frame_threshold,
            iou_threshold=self.iou_threshold,
        )
        # Format inputs as [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        detections = [
            np.hstack([box, score.T.reshape(-1, 1)])
            for box, score in zip(bboxes, scores)
        ]

        raw_tracks = []
        for detection in detections:
            raw_tracks.append(tracker.update(detection))

        dict_tracks, dict_indices = defaultdict(list), defaultdict(list)
        for frame_index, frame_detections in enumerate(raw_tracks):
            for detection in frame_detections:
                # Update the track
                track_id = detection[-1]
                bbox = detection[:-1]
                dict_tracks[track_id].append(np.hstack([frame_index, bbox]))
                # Retrieve the bounding-box index to macth it with masks
                bbox_ious = compute_bbox_iou(bboxes[frame_index], bbox)
                matching_index = np.argmax(bbox_ious)
                dict_indices[track_id].append((frame_index, matching_index))

        bbox_tracks = [
            self.interpolate_bboxes(
                np.array(track), int(track[0][0]), int(track[-1][0]) + 1
            )
            for track in dict_tracks.values()
        ]
        index_tracks = dict(dict_indices)

        return bbox_tracks, index_tracks

    def track_masks(
        self, masks: List[np.array], index_tracks: Dict[int, Tuple[int, int]]
    ) -> List[np.array]:
        """
        Track masks from a tracked indices obtained after bounding-box
        tracking.
        """
        dict_tracks = {
            track_index: [
                (frame_index, masks[frame_index][mask_index])
                for frame_index, mask_index in track
            ]
            for track_index, track in index_tracks.items()
        }

        mask_tracks = [
            self.interpolate_masks(
                np.array(track, dtype=object),
                int(track[0][0]),
                int(track[-1][0]) + 1,
            )
            for track in dict_tracks.values()
        ]

        return mask_tracks
