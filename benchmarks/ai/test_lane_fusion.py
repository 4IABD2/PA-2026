"""Unit tests for the AI-side drivable-area offset fallback (lane_fusion).

Exercises drivable_offset() directly on synthetic masks — no CARLA, no YOLOPv2
model load (LaneDetector is only imported, never instantiated)."""

import numpy as np
import pytest

from src.ai.inference.lane_fusion import drivable_offset

W, H = 1280, 720


def _mask(x0: int, x1: int) -> np.ndarray:
    """Full-frame mask with the drivable band filled between columns [x0, x1)."""
    m = np.zeros((H, W), dtype=np.uint8)
    m[int(0.80 * H) :, x0:x1] = 255  # only the bottom band is read
    return m


def test_drivable_offset_centered_is_zero():
    # drivable area symmetric around the image centre -> offset ~ 0
    off = drivable_offset(_mask(540, 740), W, H)
    assert off == pytest.approx(0.0, abs=1e-2)


def test_drivable_offset_shifted_right_is_positive():
    # drivable area to the right of centre -> car is left of the lane -> positive
    # (same sign convention as Karim's lane offset)
    off = drivable_offset(_mask(900, 1100), W, H)
    assert off > 0.3


def test_drivable_offset_shifted_left_is_negative():
    off = drivable_offset(_mask(180, 380), W, H)
    assert off < -0.3


def test_drivable_offset_none_when_empty():
    assert drivable_offset(np.zeros((H, W), dtype=np.uint8), W, H) is None


def test_drivable_offset_none_when_mask_is_none():
    assert drivable_offset(None, W, H) is None


def test_drivable_offset_clipped_to_unit_range():
    off = drivable_offset(_mask(1270, 1280), W, H)  # far right sliver
    assert -1.0 <= off <= 1.0
