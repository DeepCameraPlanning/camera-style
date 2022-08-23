import numpy as np
from typing import Tuple


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
    """Pad frame to have dimensions divisble by grid_dims."""
    frame_height, frame_width = frame.shape[:2]
    n_row, n_col = grid_dims

    padding_height = (n_row - frame_height % n_row) % n_row
    # Check if padding_height is not divisible by 2
    if padding_height % 2:
        before_padding_height = padding_height // 2
        after_padding_height = padding_height // 2 + 1
    else:
        before_padding_height = padding_height // 2
        after_padding_height = padding_height // 2

    padding_width = (n_col - frame_width % n_col) % n_col
    # Check if padding_width is not divisible by 2
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
    """Split a frame into a grid of patches according to grid_dims."""
    padded_frame = pad_frame(frame, grid_dims)

    frame_height, frame_width = padded_frame.shape[:2]
    n_row, n_col = grid_dims[0]
    y_stride, x_stride = frame_height // n_row, frame_width // n_col

    # placeholding?

    patches = [[None for _ in range(n_col)] for _ in range(n_row)]

    for i, y in enumerate(range(0, frame_height, y_stride)):
        for j, x in enumerate(range(0, frame_width, x_stride)):
            x_slice = slice(x, x + x_stride)
            y_slice = slice(y, y + y_stride)
            patches[i][j] = padded_frame[y_slice, x_slice]

    return np.array(patches)
