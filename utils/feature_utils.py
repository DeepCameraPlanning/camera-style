from typing import List, Tuple

import numpy as np
import torch
import librosa
import matplotlib.pyplot as plt

from utils.file_utils import load_pickle


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
