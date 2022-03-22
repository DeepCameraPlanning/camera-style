from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import librosa
import matplotlib.pyplot as plt

from src.utils.tracking import SortTracker
from src.utils.file_utils import load_pickle


def save_spectrogram(
    spec: torch.Tensor,
    path: str,
    title: str = None,
    ylabel: str = "freq_bin",
    aspect: str = "auto",
    xmax: float = None,
):
    """Save a spectrogram plot."""
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(spec), origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.savefig(path)


def compute_bbox_area(bboxes: np.array) -> np.array:
    """Compute areas of a set of bounding-boxes."""
    width = bboxes[:, 2] - bboxes[:, 0]
    height = bboxes[:, 3] - bboxes[:, 1]

    return width * height


def compute_bbox_overlap(bboxes: np.array, bbox: np.array) -> np.array:
    """Compute overlaps between a set of bounding-boxes and one box."""
    xmin = np.maximum(bboxes[:, 0], bbox[0])
    xmax = np.minimum(bboxes[:, 2], bbox[2])
    width = np.maximum(0, xmax - xmin)

    ymin = np.maximum(bboxes[:, 1], bbox[1])
    ymax = np.minimum(bboxes[:, 3], bbox[3])
    height = np.maximum(0, ymax - ymin)

    return width * height


def compute_bbox_iou(bboxes: np.array, bbox: np.array) -> np.array:
    """Compute IoU between a set of bounding-boxes and one box."""
    areas_1 = compute_bbox_area(bboxes)
    areas_2 = compute_bbox_area(bbox.reshape(1, -1))

    overlaps = compute_bbox_overlap(bboxes, bbox)

    iou = overlaps / (areas_1 + areas_2 - overlaps)

    return iou


def pad_frame(frame: np.array, grid_dims: Tuple[int, int]) -> np.array:
    """Pad frame to have dimensions divisble by `grid_dims`."""
    frame_height, frame_width = frame.shape[:2]
    n_row, n_col = grid_dims

    padding_height = (n_row - frame_height % n_row) % n_row
    # Check if `padding_height` is not divisible by 2
    if padding_height % 2:
        before_padding_height = padding_height // 2
        after_padding_height = padding_height // 2 + 1
    else:
        before_padding_height = padding_height // 2
        after_padding_height = padding_height // 2

    padding_width = (n_col - frame_width % n_col) % n_col
    # Check if `padding_width` is not divisible by 2
    if padding_width % 2:
        before_padding_width = padding_width // 2
        after_padding_width = padding_width // 2 + 1
    else:
        before_padding_width = padding_width // 2
        after_padding_width = padding_width // 2

    pad_dims = [
        (before_padding_height, after_padding_height),
        (before_padding_width, after_padding_width),
    ]
    if len(frame.shape) == 3:
        pad_dims.append((0, 0))

    padded_frame = np.pad(frame, pad_dims, mode="edge")

    return padded_frame


def get_patches(frame: np.array, grid_dims: Tuple[int, int]) -> np.array:
    """Split a frame into a grid of patches according to `grid_dims`."""
    padded_frame = pad_frame(frame, grid_dims)

    frame_height, frame_width = padded_frame.shape[:2]
    n_row, n_col = grid_dims
    y_stride, x_stride = frame_height // n_row, frame_width // n_col

    patches = [[None for _ in range(n_col)] for _ in range(n_row)]
    for i, y in enumerate(range(0, frame_height, y_stride)):
        for j, x in enumerate(range(0, frame_width, x_stride)):
            x_slice = slice(x, x + x_stride)
            y_slice = slice(y, y + y_stride)
            patches[i][j] = padded_frame[y_slice, x_slice]

    return np.array(patches)


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
