import math
import cv2
import numpy as np

MIN_LANE_W = 120  # largeur mini (px) d'une voie ego plausible en bas d'image
MAX_LANE_FRAC = 0.55  # largeur maxi (frac w) d'une voie ego
DEFAULT_LANE_FRAC = 0.30  # largeur de voie supposee si un seul bord detecte
EGO_BAND_FRAC = 0.90  # bande basse utilisee pour identifier les lignes ego
CLUSTER_GAP = 20  # ecart (px) qui separe deux lignes distinctes
TRACK_WIN_FRAC = 0.06  # demi-fenetre (frac w) de suivi d'un bord vers le haut
TOP_FRAC = 0.45  # on arrete l'analyse a cette hauteur (horizon)
DASH_BRIDGE = 25  # fermeture verticale (px) : relie les pointilles pour le suivi
LOOKAHEAD_FRAC = 0.50  # point vise : fraction du bas (0) vers le haut (1)
ANGLE_DEADZONE = 3.0  # |angle| (deg) en dessous duquel on considere "ALIGNE"


def _lane_clusters(track, w, h):
    """Centres des lignes (clusters de colonnes) sur une bande basse de l'image."""
    band = track[int(EGO_BAND_FRAC * h) :, :]
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
    """2 bords de la voie ego : la paire de lignes qui encadre la voiture avec une
    largeur plausible et dont le centre est le plus proche du centre camera."""
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

    # un seul bord detecte : on derive l'autre a une largeur de voie par defaut
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

    # relie verticalement les pointilles pour le SUIVI (l'affichage garde l'original)
    track = cv2.morphologyEx(
        lanes,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, DASH_BRIDGE)),
    )

    ego = _ego_init(track, w, h)
    if ego is None:
        return result
    left_x, right_x = ego

    # suivi des 2 bords vers le haut, en collectant les points (la ou il y en a)
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

    # une droite par cote : l'ajustement traverse les trous des pointilles
    fl = np.polyfit(lys, lxs, 1)
    fr = np.polyfit(rys, rxs, 1)
    ys_line = np.linspace(h - 1, int(TOP_FRAC * h), 30)  # du bas vers le haut
    cxs = 0.5 * (np.polyval(fl, ys_line) + np.polyval(fr, ys_line))
    center = np.stack([cxs, ys_line], axis=1)
    result["center_line"] = center.astype(np.int32)

    # point vise (pour l'angle de braquage — mesure au lookahead, inchange)
    idx = int(np.clip(round(LOOKAHEAD_FRAC * (len(center) - 1)), 0, len(center) - 1))
    aim_x, aim_y = center[idx]
    angle = math.degrees(math.atan2(aim_x - w / 2.0, (h - 1) - aim_y))

    # offset (pour r_center — mesure a la position de la voiture, bas de l'image,
    # normalise par la vraie largeur de voie a cette hauteur, pas par la demi-
    # largeur d'image : ±1 doit vouloir dire "bord de la voie", pas "bord de
    # l'image")
    bottom_y = h - 1.0
    lane_left_x = np.polyval(fl, bottom_y)
    lane_right_x = np.polyval(fr, bottom_y)
    lane_center_x = 0.5 * (lane_left_x + lane_right_x)
    lane_half_width = max((lane_right_x - lane_left_x) / 2.0, 1.0)  # garde div/0
    offset = (lane_center_x - w / 2.0) / lane_half_width
    direction = (
        "DROITE"
        if angle > ANGLE_DEADZONE
        else "GAUCHE" if angle < -ANGLE_DEADZONE else "ALIGNE"
    )

    # trajectoire : de la voiture (bas, centre) vers le point vise puis le centre
    car = np.array([w / 2.0, h - 1.0])
    target = np.array([aim_x, aim_y])
    ctrl = np.array([w / 2.0, (car[1] + target[1]) / 2.0])
    ts = np.linspace(0, 1, 16)[:, None]
    bez = (1 - ts) ** 2 * car + 2 * (1 - ts) * ts * ctrl + ts**2 * target
    traj = np.vstack([bez, center[idx:]])

    result.update(
        {
            "status": "OK",
            "trajectory": traj.astype(np.int32),
            "offset": float(offset),
            "direction": direction,
            "angle": float(angle),
        }
    )
    return result
