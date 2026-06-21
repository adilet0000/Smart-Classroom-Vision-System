"""
Pure-numpy geometry helpers for bounding boxes and crops.

These are deliberately dependency-light (numpy only) so they can be unit-tested
without importing the heavy detection / recognition stack (torch, insightface,
mediapipe).  Both the person detector and the face embedder share this logic to
guarantee consistent clamping and cropping behaviour.
"""
from __future__ import annotations

import numpy as np


def clamp_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """
    Clamp a box to the frame bounds and guarantee x1<=x2, y1<=y2.

    Detector outputs can sit slightly outside the frame (negative coords or
    coords past width/height); slicing with those produces empty or wrapped
    crops.  This normalises the box to valid integer pixel coordinates.
    """
    x1 = int(max(0, min(x1, width)))
    y1 = int(max(0, min(y1, height)))
    x2 = int(max(0, min(x2, width)))
    y2 = int(max(0, min(y2, height)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def is_valid_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    min_width: int = 0,
    min_height: int = 0,
    min_area: int = 0,
) -> bool:
    """
    Reject degenerate / tiny boxes that produce useless crops.

    A box passes when it is at least ``min_width`` × ``min_height`` pixels and
    its area is at least ``min_area``.  Zero-area boxes always fail.
    """
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return False
    if w < min_width or h < min_height:
        return False
    if w * h < min_area:
        return False
    return True


def crop_head_region(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    top_ratio: float = 0.55,
    side_padding_ratio: float = 0.08,
) -> np.ndarray:
    """
    Crop the upper portion of a person box (where the head/face is) with a small
    horizontal padding.  Always clamped to the frame, so it never raises on
    out-of-range boxes and returns an empty array for degenerate input.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box

    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = int(box_w * side_padding_ratio)

    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1)
    crop_x2 = min(w, x2 + pad_x)
    crop_y2 = min(h, y1 + int(box_h * top_ratio))

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return frame[0:0, 0:0]

    return frame[crop_y1:crop_y2, crop_x1:crop_x2]
