"""Lane-signal fusion (AI side) — adds a drivable-area fallback offset on top of
Karim's YOLOPv2 lane detector, WITHOUT modifying his module.

Why: on Town02 the painted lane lines are so sparse that
`lane_perception.estimate()` returns direction="NONE" ~94% of the time, so the
policy's lateral-offset input goes stale and it drives off-centre (see v22).
YOLOPv2 *also* emits a drivable-area mask that is present ~98% of the time; its
horizontal centroid is a fresh lateral-position signal (validated: r=-0.57 vs
CARLA ground truth, versus r=-0.18 for the lane-line offset when it does fire).

We only *read* the `drivable` mask Karim's detector already produces and turn it
into an offset here. His `LaneDetector` / `lane_geometry` code is untouched; this
stays 0 ground-truth (the mask is real perception).
"""

from __future__ import annotations

import numpy as np

import src.lane_detection.lane_perception as _lp
from src.lane_detection.lane_perception import LaneDetector  # Karim's, unmodified

_DRIVABLE_BAND_FRAC = 0.80  # bottom 20% of the frame (closest to the car)
_DRIVABLE_MIN_COLS = 5  # need at least this many drivable columns to trust it


def drivable_offset(drivable, w: int, h: int) -> float | None:
    """Lateral position from YOLOPv2's drivable-area mask: mass-weighted
    horizontal centroid of the drivable pixels in the bottom band vs image
    centre, normalised to [-1, 1] (same sign convention as Karim's lane offset:
    negative = drivable area sits left, i.e. the car is right of centre).
    Returns None when the band holds too little drivable area to be trusted."""
    if drivable is None:
        return None
    band = drivable[int(_DRIVABLE_BAND_FRAC * h) :, :]
    col_mass = band.sum(axis=0)
    cols = np.where(col_mass > 0)[0]
    if len(cols) < _DRIVABLE_MIN_COLS:
        return None
    cx = float((cols * col_mass[cols]).sum() / col_mass[cols].sum())
    return float(np.clip((cx - w / 2.0) / (w / 2.0), -1.0, 1.0))


def estimate_with_drivable(rgb, detector: LaneDetector | None = None):
    """Drop-in for `lane_perception.estimate` that ALSO returns the drivable-area
    fallback offset. Output: (direction, angle, offset, drivable_offset), where
    drivable_offset is None when unavailable. Reuses Karim's detector singleton
    (his model, one forward pass) — we just also read its `drivable` mask."""
    if detector is None:
        if _lp._DETECTOR is None:
            _lp._DETECTOR = LaneDetector()
        detector = _lp._DETECTOR
    r = detector.detect(rgb)
    h, w = rgb.shape[:2]
    return r["direction"], r["angle"], r["offset"], drivable_offset(r["drivable"], w, h)
