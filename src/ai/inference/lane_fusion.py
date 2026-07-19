"""Lane offset with a drivable-area fallback.

The YOLOPv2 lane-line detector often finds no painted lines on Town02, but its
drivable-area mask is almost always available. When the lines are missing, the
lateral offset is derived from that mask instead, so the policy always gets a
fresh reading. lane_perception itself is not modified.
"""

import numpy as np

from src.lane_detection.lane_perception import LaneDetector

_BAND_FRAC = 0.80  # bottom part of the image used for the offset
_MIN_COLS = 5  # minimum drivable columns needed to trust the estimate

_detector = None


def drivable_offset(drivable, w: int, h: int) -> float | None:
    """Lateral position from the drivable-area mask, in [-1, 1] (0 = centred).

    Weighted horizontal centroid of the mask's bottom band vs the image centre,
    same sign convention as the lane-line offset. Returns None when there is
    not enough drivable area to measure.
    """
    if drivable is None:
        return None
    band = drivable[int(_BAND_FRAC * h) :, :]
    col_mass = band.sum(axis=0)
    cols = np.where(col_mass > 0)[0]
    if len(cols) < _MIN_COLS:
        return None
    cx = float((cols * col_mass[cols]).sum() / col_mass[cols].sum())
    return float(np.clip((cx - w / 2.0) / (w / 2.0), -1.0, 1.0))


def estimate_with_drivable(rgb):
    """Like lane_perception.estimate, plus the drivable-area fallback offset.

    Returns (direction, angle, offset, drivable_offset), where drivable_offset
    is None when unavailable. The detector is loaded once (module singleton).
    """
    global _detector
    if _detector is None:
        _detector = LaneDetector()
    r = _detector.detect(rgb)
    h, w = rgb.shape[:2]
    return r["direction"], r["angle"], r["offset"], drivable_offset(r["drivable"], w, h)
