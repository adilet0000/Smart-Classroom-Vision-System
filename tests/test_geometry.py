"""Bounding-box clamp / validity / crop edge cases (pure numpy, no ML stack)."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.geometry import clamp_box, crop_head_region, is_valid_box  # noqa: E402


def test_clamp_box_keeps_in_bounds_box_unchanged():
    assert clamp_box(10, 20, 100, 200, 640, 480) == (10, 20, 100, 200)


def test_clamp_box_clips_negative_and_overflow():
    # Negative origin and coords past the frame are pulled back inside.
    assert clamp_box(-50, -30, 700, 600, 640, 480) == (0, 0, 640, 480)


def test_clamp_box_swaps_inverted_coords():
    assert clamp_box(200, 300, 100, 50, 640, 480) == (100, 50, 200, 300)


def test_is_valid_box_rejects_tiny():
    assert is_valid_box(0, 0, 10, 10, min_width=20, min_height=40) is False


def test_is_valid_box_rejects_zero_area():
    assert is_valid_box(5, 5, 5, 5) is False
    assert is_valid_box(5, 5, 5, 80) is False  # zero width


def test_is_valid_box_accepts_normal_person():
    assert is_valid_box(0, 0, 60, 160, min_width=20, min_height=40) is True


def test_is_valid_box_min_area():
    # 30x30 = 900 area; require 2000 -> reject.
    assert is_valid_box(0, 0, 30, 30, min_area=2000) is False
    assert is_valid_box(0, 0, 60, 60, min_area=2000) is True


def test_crop_head_region_normal():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    crop = crop_head_region(frame, (100, 100, 200, 300), top_ratio=0.55, side_padding_ratio=0.08)
    # height = 200 * 0.55 = 110; width = 100 + 2*8 padding = 116
    assert crop.shape[0] == 110
    assert crop.shape[1] == 116
    assert crop.size > 0


def test_crop_head_region_out_of_frame_is_clamped_not_crashing():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    crop = crop_head_region(frame, (600, 400, 900, 900))
    # Clamped to the frame, still a valid (possibly small) crop, never raises.
    assert crop.ndim == 3
    assert crop.shape[1] <= 640


def test_crop_head_region_degenerate_returns_empty():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    crop = crop_head_region(frame, (300, 300, 300, 300))
    assert crop.size == 0


def test_crop_head_region_fully_offscreen_returns_empty():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Box entirely to the right of the frame -> nothing to crop.
    crop = crop_head_region(frame, (700, 100, 800, 300))
    assert crop.size == 0


if __name__ == "__main__":
    import sys

    from tests.run_tests import run_module

    sys.exit(0 if run_module(sys.modules[__name__]) else 1)
