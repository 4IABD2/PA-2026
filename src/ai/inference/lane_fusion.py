"""Lane offset with a drivable-area fallback."""

import numpy as np

from src.lane_detection.lane_perception import LaneDetector

_BAND_FRAC = 0.80
_MIN_COLS = 5

_detector = None


def drivable_offset(drivable, w: int, h: int) -> float | None:
    """Lateral offset from the drivable-area mask, in [-1, 1] (0 = centred)."""
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
    """Like lane_perception.estimate, plus the drivable-area fallback offset."""
    global _detector
    if _detector is None:
        _detector = LaneDetector()
    r = _detector.detect(rgb)
    h, w = rgb.shape[:2]
    return r["direction"], r["angle"], r["offset"], drivable_offset(r["drivable"], w, h)
