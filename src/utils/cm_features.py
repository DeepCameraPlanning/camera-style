from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch

from src.utils.tracking import SortTracker
from src.utils.utils import load_pickle


def apply_nms(bboxes: torch.Tensor, threshold_iou: float):
    """
    Code adapted from `https://learnopencv.com/non-maximum-suppression-theory-
    and-implementation-in-pytorch/`.
    Apply non-maximum suppression to avoid detecting too many overlapping
    bounding boxes for a given object.

    :param boxes: location preds for the image along with the class predscores,
        dim: (num_boxes, 5).
    :param threshold_iou: overlap threshold for suppressing unnecessary boxes.
    :return: a list of filtered boxes, dim: (, 5).
    """
    # Extract coordinates and confidence scores for every bboxes
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    scores = bboxes[:, 4]

    # Calculate area of every block in P
    areas = (x2 - x1) * (y2 - y1)
    # Sort the prediction boxes according to their confidence scores
    order = scores.argsort()

    filtered_bboxes = []
    while len(order) > 0:
        # Extract the index of the prediction with highest scores
        top_pred_index = order[-1]
        filtered_bboxes.append(bboxes[top_pred_index].numpy())
        order = order[:-1]

        # Check if there are still predictions to process
        if len(order) == 0:
            break

        # Select coordinates of bboxes according to the indices in order
        xx1 = torch.index_select(x1, dim=0, index=order)
        xx2 = torch.index_select(x2, dim=0, index=order)
        yy1 = torch.index_select(y1, dim=0, index=order)
        yy2 = torch.index_select(y2, dim=0, index=order)
        # Find the coordinates of the intersection boxes
        xx1 = torch.max(xx1, x1[top_pred_index])
        yy1 = torch.max(yy1, y1[top_pred_index])
        xx2 = torch.min(xx2, x2[top_pred_index])
        yy2 = torch.min(yy2, y2[top_pred_index])
        # Find height and width of the intersection boxes
        w = xx2 - xx1
        h = yy2 - yy1
        # Avoid negative w and h due to non-overlapping boxes
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)

        # Find intersection area
        inter = w * h
        # Find areas of bboxes according the indices in order and union area
        rem_areas = torch.index_select(areas, dim=0, index=order)
        union = (rem_areas - inter) + areas[top_pred_index]
        # Compute the IoU
        IoU = inter / union

        # Keep the boxes with IoU less than `threshold_iou``
        mask = IoU < threshold_iou
        order = order[mask]

    return filtered_bboxes


def track_bboxes(
    preprocessed_detections: List[np.array], shot_boundaries: List[int]
) -> List[np.array]:
    """Track bounding-boxes within each shot."""
    n_frames = shot_boundaries[-1] + 1
    preprocessed_detections1 = [
        np.array(preprocessed_detections[frame_index])
        if len(preprocessed_detections[frame_index]) > 0
        else np.empty((0, 5))
        for frame_index in range(n_frames + 1)
    ]

    preprocessed_tracks = []
    start_boundary = 0
    for end_boundary in shot_boundaries:
        end_boundary += 1
        # Get bbox of the current shot
        shot_detections = preprocessed_detections1[start_boundary:end_boundary]
        # Track the different bboxes
        shot_tracker = SortTracker()
        shot_tracks = shot_tracker.track_bboxes(shot_detections)
        # Add the frame index offset
        for track in shot_tracks:
            track[:, 0] += start_boundary

        preprocessed_tracks.extend(shot_tracks)
        start_boundary = end_boundary

    return preprocessed_tracks


def preprocess_bodytracks(
    raw_bodytracks: List[Any],
    shot_boundaries: List[int],
    threshold_iou: float,
    current_fps: float,
) -> List[np.array]:
    """
    Preprocess precomputed Condensed Movies body-tracks features, by applying
    NMS on each frame detection. Detections are adapted to 25 fps.

    :param raw_bodytracks: raw list of precomputed body-tracks features.
    :param threshold_iou: NMS IoU threshold.
    :param current_fps: fps of the current clip.
    :return: a map between frame index and normalized bboxes.
    """
    target_fps = 25

    # Initialize a dict of lists to return an empty list when a frame misses
    preprocessed_detections = defaultdict(list)
    RR_detect = defaultdict(list)
    for raw_subtracks in raw_bodytracks:
        for raw_detections in raw_subtracks.frame_based_indexing.items():
            # Add +2 empirically (not understood why)
            frame_index = int(target_fps * raw_detections[0] / current_fps) + 2
            raw_bboxes = torch.from_numpy(raw_detections[1][0])
            preprocessed_bboxes = apply_nms(raw_bboxes, threshold_iou)
            preprocessed_detections[frame_index].extend(preprocessed_bboxes)
            RR_detect[raw_detections[0]].extend(preprocessed_bboxes)

    preprocessed_bodytracks = track_bboxes(
        preprocessed_detections, shot_boundaries
    )

    return preprocessed_bodytracks


def preprocess_facetracks(
    database: Dict[str, Any],
    raw_detections: np.array,
    shot_boundaries: List[int],
    current_fps: float,
) -> List[np.array]:
    """
    Preprocess precomputed Condensed Movies face-tracks features. Detections
    are made with clips of shape 640x360 and adapted to 25 fps.

    :param database: face detection metadata.
    :param raw_detections: raw face detections.
    :param shot_boundaries: list of shot boundaries.
    :param current_fps: fps of the current clip.
    :return: a map between frame index and normalized bboxes.
    """
    target_fps = 25
    target_width, target_height = 640, 360

    # Initialize a dict of lists to return an empty list when a frame misses
    preprocessed_detections = defaultdict(list)
    for face_track_frames in database["index_into_facedetfile"]:
        for index in face_track_frames:
            frame_index, x, y, dx, dy, _, score = raw_detections[index]
            frame_index = int(
                target_fps * raw_detections[index][0] / current_fps
            )
            preprocessed_bbox = np.array(
                [
                    x / target_width,
                    y / target_height,
                    (x + dx) / target_width,
                    (y + dy) / target_height,
                    score,
                ]
            )
            preprocessed_detections[frame_index].append(preprocessed_bbox)

    preprocessed_facetracks = track_bboxes(
        preprocessed_detections, shot_boundaries
    )

    return preprocessed_facetracks


def load_framebboxes(track_path: str, n_frames: int) -> List[List[np.array]]:
    """Load the list of bboxes per frame, instead of per track."""
    tracks = load_pickle(track_path)
    frame_bboxes = []
    for frame_index in range(n_frames):
        current_framebboxes = []
        for t in tracks:
            (box_index,) = np.where(t[:, 0] == frame_index)
            bbox = t[box_index].reshape(-1)
            if bbox.size > 0:
                current_framebboxes.append(bbox[1:5])

        if len(current_framebboxes) == 0:
            current_framebboxes = np.empty((0, 4))
        frame_bboxes.append(current_framebboxes)

    return frame_bboxes
