"""Tests for lane_geometry()'s offset computation.

offset is one of the three values src.lane_detection.lane_perception.estimate()
returns, and it becomes lane_offset_norm in the RL observation / drives the
r_center reward term. It must be measured at the car's actual position (the
bottom row of the image) and normalised by the real lane half-width there --
NOT at a lookahead point partway up the image, normalised by half the image
width. angle (and therefore the steering command) must be completely
unaffected by this change.

The synthetic `lanes` masks below are two straight lane-edge lines drawn with
cv2.line (the same import lane_geometry.py already uses) spanning the full
image height. Because the geometry is fully known ahead of time, the expected
fl/fr polyfits -- and therefore the expected offset/angle -- can be derived
analytically instead of guessed at.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from src.lane_detection.lane_geometry import LOOKAHEAD_FRAC, TOP_FRAC, lane_geometry

W, H = 800, 400


def _draw_edge(
    mask: np.ndarray, x_bottom: float, x_top: float, thickness: int = 3
) -> None:
    """Draw one straight lane-edge line spanning the full image height, defined
    by its x-position at the bottom row (y = H-1, the car's row) and the top
    row (y = 0). A straight line drawn this way is recovered near-exactly by
    np.polyfit(degree=1), so the drawn (x_bottom, x_top) fully determine the
    fl/fr fits lane_geometry() will compute.
    """
    cv2.line(
        mask,
        (int(round(x_bottom)), H - 1),
        (int(round(x_top)), 0),
        color=255,
        thickness=thickness,
    )


def _make_lanes(
    left_bottom: float, left_top: float, right_bottom: float, right_top: float
) -> np.ndarray:
    lanes = np.zeros((H, W), dtype=np.uint8)
    _draw_edge(lanes, left_bottom, left_top)
    _draw_edge(lanes, right_bottom, right_top)
    return lanes


def _drivable() -> np.ndarray:
    # lane_geometry() stores this verbatim in the result dict but never reads
    # it -- a same-shape dummy is enough.
    return np.zeros((H, W), dtype=np.uint8)


def _line_params(x_bottom: float, x_top: float) -> tuple[float, float]:
    """(a, b) of x = a*y + b for a line defined by its endpoints at y=H-1
    (bottom) and y=0 (top) -- the same two points used to draw it, so this is
    the exact fit np.polyfit should recover (up to pixel-rounding noise)."""
    a = (x_bottom - x_top) / (H - 1)
    b = x_top
    return a, b


def _expected_lookahead_aim_and_angle(
    left_bottom: float, left_top: float, right_bottom: float, right_top: float
) -> tuple[float, float, float]:
    """Reproduces, purely analytically from the drawn geometry, the lookahead
    aim point (aim_x, aim_y) and the steering angle derived from it -- i.e.
    exactly what the pre-existing (unchanged) angle computation in
    lane_geometry() does. Used both to hand-check that angle is unaffected by
    the offset fix, and to demonstrate what the OLD offset computation
    (aim_x-based) would have produced.
    """
    a_l, b_l = _line_params(left_bottom, left_top)
    a_r, b_r = _line_params(right_bottom, right_top)
    ys_line = np.linspace(H - 1, int(TOP_FRAC * H), 30)
    idx = int(np.clip(round(LOOKAHEAD_FRAC * (len(ys_line) - 1)), 0, len(ys_line) - 1))
    aim_y = ys_line[idx]
    aim_x = 0.5 * ((a_l * aim_y + b_l) + (a_r * aim_y + b_r))
    angle = math.degrees(math.atan2(aim_x - W / 2.0, (H - 1) - aim_y))
    return aim_x, aim_y, angle


def test_straight_centered_lane_offset_is_zero():
    """Case 1: both edges vertical (no curve), lane centered under the car."""
    lanes = _make_lanes(
        left_bottom=300.0, left_top=300.0, right_bottom=500.0, right_top=500.0
    )

    result = lane_geometry(lanes, _drivable(), W, H)

    assert result["status"] == "OK"
    assert result["offset"] == pytest.approx(0.0, abs=0.01)


def test_curving_lane_still_centered_at_car_offset_is_zero():
    """Case 2 (regression test for the lookahead-location bug): the lane
    widens sharply toward the horizon -- edges have different, nonzero slopes
    -- but stays perfectly centered under the car at the bottom row
    ((left_bottom + right_bottom) / 2 == w / 2). The OLD code measured offset
    at the lookahead point, where this geometry produces a clearly nonzero
    raw offset (asserted below via the analytic aim_x); the NEW code measures
    offset at the car's row, where it must be ~0.
    """
    left_bottom, left_top = 300.0, 281.78
    right_bottom, right_top = 500.0, 773.28
    assert (left_bottom + right_bottom) / 2.0 == W / 2.0  # centered at the car's row
    lanes = _make_lanes(left_bottom, left_top, right_bottom, right_top)

    result = lane_geometry(lanes, _drivable(), W, H)

    assert result["status"] == "OK"
    assert result["offset"] == pytest.approx(0.0, abs=0.03)

    # Prove this geometry is a real regression case: the OLD (lookahead-based,
    # image-width-normalised) measurement would NOT have been ~0 here.
    aim_x, _, _ = _expected_lookahead_aim_and_angle(
        left_bottom, left_top, right_bottom, right_top
    )
    old_offset = (aim_x - W / 2.0) / (W / 2.0)
    assert abs(old_offset) > 0.05


def test_offset_scales_with_real_lane_width_not_image_width():
    """Case 3 (regression test for the normalisation bug): two lanes with the
    SAME raw pixel deviation of the car from the lane's center-x, but
    different real widths. The narrower lane must show a bigger normalized
    offset for the same pixel deviation -- today (normalised by w/2 for both)
    they would be identical.
    """
    same_pixel_deviation = 30.0
    center_x = W / 2.0 + same_pixel_deviation

    narrow_half_width = 100.0  # 200px-wide lane
    wide_half_width = 200.0  # 400px-wide lane

    narrow_lanes = _make_lanes(
        center_x - narrow_half_width,
        center_x - narrow_half_width,
        center_x + narrow_half_width,
        center_x + narrow_half_width,
    )
    wide_lanes = _make_lanes(
        center_x - wide_half_width,
        center_x - wide_half_width,
        center_x + wide_half_width,
        center_x + wide_half_width,
    )

    narrow_result = lane_geometry(narrow_lanes, _drivable(), W, H)
    wide_result = lane_geometry(wide_lanes, _drivable(), W, H)

    assert narrow_result["status"] == "OK"
    assert wide_result["status"] == "OK"

    expected_narrow = same_pixel_deviation / narrow_half_width
    expected_wide = same_pixel_deviation / wide_half_width

    assert narrow_result["offset"] == pytest.approx(expected_narrow, abs=0.01)
    assert wide_result["offset"] == pytest.approx(expected_wide, abs=0.01)
    assert narrow_result["offset"] > wide_result["offset"]


def test_angle_unaffected_by_offset_fix():
    """Case 4: angle must still be derived from the lookahead point exactly as
    before -- the offset fix must not have touched it. Uses the same curving
    geometry as case 2, since it produces a non-trivial (nonzero) angle."""
    left_bottom, left_top = 300.0, 281.78
    right_bottom, right_top = 500.0, 773.28
    lanes = _make_lanes(left_bottom, left_top, right_bottom, right_top)

    result = lane_geometry(lanes, _drivable(), W, H)

    assert result["status"] == "OK"
    _, _, expected_angle = _expected_lookahead_aim_and_angle(
        left_bottom, left_top, right_bottom, right_top
    )
    assert result["angle"] == pytest.approx(expected_angle, abs=1.0)
