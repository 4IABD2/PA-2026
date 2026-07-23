import math
import cv2
import numpy as np

MIN_LANE_W = 120
MAX_LANE_FRAC = 0.55
DEFAULT_LANE_FRAC = 0.30
EGO_BAND_FRAC = 0.90
CLUSTER_GAP = 20
TRACK_WIN_FRAC = 0.06
TOP_FRAC = 0.45
DASH_BRIDGE = 25
LOOKAHEAD_FRAC = 0.50
ANGLE_DEADZONE = 3.0


def _lane_clusters(track, w, h):
    band = track[int(EGO_BAND_FRAC * h):, :]
    cols = np.where(band.sum(axis=0) > 0)[0]
    if len(cols) == 0:
        return np.array([])

    clusters, start, prev = [], cols[0], cols[0]
    for c in cols[1:]:
        if c - prev > CLUSTER_GAP:
            clusters.append((start + prev) / 2.0)
            start = c
        prev = c
    clusters.append((start + prev) / 2.0)
    return np.array(clusters)


def _ego_init(track, w, h):
    clusters = _lane_clusters(track, w, h)
    if len(clusters) == 0:
        return None

    left = clusters[clusters < w / 2.0]
    right = clusters[clusters > w / 2.0]

    best = None
    for li in left:
        for rj in right:
            width = rj - li
            if MIN_LANE_W <= width <= MAX_LANE_FRAC * w:
                err = abs(0.5 * (li + rj) - w / 2.0)
                if best is None or err < best[0]:
                    best = (err, float(li), float(rj))
    if best:
        return best[1], best[2]

    near = float(clusters[np.argmin(np.abs(clusters - w / 2.0))])
    dw = DEFAULT_LANE_FRAC * w
    return (near, near + dw) if near < w / 2.0 else (near - dw, near)


def lane_geometry(lanes, drivable, w, h):
    result = {
        "status": "NO_LANE",
        "drivable": drivable,
        "lanes": lanes,
        "center_line": None,
        "trajectory": None,
        "offset": 0.0,
        "direction": "NONE",
        "angle": 0.0,
    }

    track = cv2.morphologyEx(
        lanes,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, DASH_BRIDGE)),
    )

    ego = _ego_init(track, w, h)
    if ego is None:
        return result
    left_x, right_x = ego

    win = TRACK_WIN_FRAC * w
    lxs, lys, rxs, rys = [], [], [], []
    for y in range(h - 1, int(TOP_FRAC * h), -max(1, h // 70)):
        xs = np.where(track[y] > 0)[0]
        if len(xs):
            cl = xs[np.abs(xs - left_x) < win]
            cr = xs[np.abs(xs - right_x) < win]
            if len(cl):
                left_x = float(np.median(cl))
                lxs.append(left_x)
                lys.append(float(y))
            if len(cr):
                right_x = float(np.median(cr))
                rxs.append(right_x)
                rys.append(float(y))

    if len(lxs) < 4 or len(rxs) < 4:
        return result

    fl = np.polyfit(lys, lxs, 1)
    fr = np.polyfit(rys, rxs, 1)
    ys_line = np.linspace(h - 1, int(TOP_FRAC * h), 30)
    cxs = 0.5 * (np.polyval(fl, ys_line) + np.polyval(fr, ys_line))
    center = np.stack([cxs, ys_line], axis=1)
    result["center_line"] = center.astype(np.int32)

    idx = int(np.clip(round(LOOKAHEAD_FRAC * (len(center) - 1)), 0, len(center) - 1))
    aim_x, aim_y = center[idx]
    angle = math.degrees(math.atan2(aim_x - w / 2.0, (h - 1) - aim_y))

    # offset normalise par la largeur reelle de la voie : ±1 = bord de la voie
    bottom_y = h - 1.0
    lane_left_x = np.polyval(fl, bottom_y)
    lane_right_x = np.polyval(fr, bottom_y)
    lane_center_x = 0.5 * (lane_left_x + lane_right_x)
    lane_half_width = max((lane_right_x - lane_left_x) / 2.0, 1.0)
    offset = (lane_center_x - w / 2.0) / lane_half_width

    direction = (
        "DROITE" if angle > ANGLE_DEADZONE else
        "GAUCHE" if angle < -ANGLE_DEADZONE else
        "ALIGNE"
    )

    car = np.array([w / 2.0, h - 1.0])
    target = np.array([aim_x, aim_y])
    ctrl = np.array([w / 2.0, (car[1] + target[1]) / 2.0])
    ts = np.linspace(0, 1, 16)[:, None]
    bez = (1 - ts) ** 2 * car + 2 * (1 - ts) * ts * ctrl + ts ** 2 * target
    traj = np.vstack([bez, center[idx:]])

    result.update({
        "status": "OK",
        "trajectory": traj.astype(np.int32),
        "offset": float(offset),
        "direction": direction,
        "angle": float(angle),
    })
    return result