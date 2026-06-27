"""Detection de voie par modele pre-entraine YOLOPv2 (deep learning, RGB pur).

YOLOPv2 sort en une passe, depuis une image RGB :
  - la ZONE ROULABLE (drivable area)
  - les LIGNES DE VOIE
On en deduit :
  - le trace des lignes de voie,
  - la trajectoire a suivre (axe central de la zone roulable),
  - la consigne d'alignement gauche/droite.

Le modele est un TorchScript autonome (un seul fichier de poids, aucune autre
dependance que torch/torchvision). Aucune fonction CARLA n'est utilisee.

Poids : telecharges automatiquement au 1er lancement dans weights/yolopv2.pt
(repo officiel CAIC-AD/YOLOPv2). Tourne sur GPU si dispo, sinon CPU (plus lent).
"""

import os
import urllib.request

import cv2
import numpy as np

try:
    import torch
except ImportError:
    torch = None

WEIGHTS_URL = "https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt"
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "yolopv2.pt")

INF_W, INF_H = 640, 480          # entree du modele (multiple de 32, ratio 4:3)
MIN_LANE_W = 120                 # largeur mini (px) d'une voie ego plausible en bas
MAX_LANE_FRAC = 0.55             # largeur maxi (frac w) d'une voie ego
EGO_BAND_FRAC = 0.90             # bande basse pour identifier les lignes ego
CLUSTER_GAP = 20                 # ecart (px) qui separe deux lignes distinctes
ALIGN_DEADZONE = 0.06
LOOKAHEAD_FRAC = 0.50            # point vise : fraction du bas vers le haut (0=bas, 1=haut)


def _ensure_weights():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    if not os.path.exists(WEIGHTS_PATH):
        print(f"[YOLOPv2] telechargement des poids -> {WEIGHTS_PATH} ...")
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
        print("[YOLOPv2] poids OK")
    return WEIGHTS_PATH


def _lane_clusters(track, w, h):
    """Centres des lignes (clusters de colonnes) sur une bande basse de l'image."""
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
    """Identifie les 2 bords de la VOIE EGO en bas : parmi toutes les paires de
    lignes qui encadrent la voiture (w/2) avec une largeur plausible, on prend
    celle dont le centre est le plus proche du centre camera (la voiture roule
    dans SA voie, centree) -> evite que le centre se pose sur une ligne."""
    clusters = _lane_clusters(track, w, h)
    left = clusters[clusters < w / 2.0]
    right = clusters[clusters > w / 2.0]
    if len(left) == 0 or len(right) == 0:
        return None
    best = None
    for li in left:
        for rj in right:
            width = rj - li
            if MIN_LANE_W <= width <= MAX_LANE_FRAC * w:
                err = abs(0.5 * (li + rj) - w / 2.0)
                if best is None or err < best[0]:
                    best = (err, float(li), float(rj))
    return (best[1], best[2]) if best else None


def build_lane_result(lanes, drivable, w, h):
    """A partir des masques 'lignes de voie' + 'zone roulable', construit le
    centre de la VOIE EGO et la trajectoire d'alignement. On identifie les deux
    bords de la voie ego en bas (paire la plus centree sur la voiture), on suit
    chacun vers le haut puis on ajuste une droite par cote (robuste aux
    pointilles). Partage modele pre-entraine / fine-tune."""
    result = {"status": "NO_LANE", "drivable": drivable, "lanes": lanes,
              "center_line": None, "trajectory": None,
              "offset": 0.0, "direction": "NONE"}

    # relie verticalement les pointilles pour le SUIVI (l'affichage 'lanes'
    # garde le masque original)
    track = cv2.morphologyEx(lanes, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (3, 25)))

    # 1. bords de la voie ego (paire de lignes la plus centree sur la voiture)
    ego = _ego_init(track, w, h)
    if ego is None:
        return result
    left_x, right_x = ego
    y_start = h - 1

    # 2. suivi des 2 lignes vers le haut en COLLECTANT les points de chaque bord
    # (seulement la ou il y a des pixels), puis on AJUSTE une droite par cote.
    # L'ajustement traverse les trous des pointilles -> le centre ne penche plus
    # vers le cote pointille.
    win = 0.06 * w
    lxs, lys, rxs, rys = [], [], [], []
    for y in range(y_start, int(0.45 * h), -max(1, h // 70)):
        xs = np.where(track[y] > 0)[0]
        if len(xs):
            cl = xs[np.abs(xs - left_x) < win]
            cr = xs[np.abs(xs - right_x) < win]
            if len(cl):
                left_x = float(np.median(cl)); lxs.append(left_x); lys.append(float(y))
            if len(cr):
                right_x = float(np.median(cr)); rxs.append(right_x); rys.append(float(y))
    if len(lxs) < 4 or len(rxs) < 4:
        return result

    fl = np.polyfit(lys, lxs, 1)               # droite par cote (degre 1, stable)
    fr = np.polyfit(rys, rxs, 1)
    ys_line = np.linspace(h - 1, int(0.45 * h), 30)        # du bas vers le haut
    cxs = 0.5 * (np.polyval(fl, ys_line) + np.polyval(fr, ys_line))
    center = np.stack([cxs, ys_line], axis=1)
    result["center_line"] = center.astype(np.int32)

    idx = int(np.clip(round(LOOKAHEAD_FRAC * (len(center) - 1)), 0, len(center) - 1))
    aim_x, aim_y = center[idx]
    offset = (aim_x - w / 2.0) / (w / 2.0)
    direction = ("DROITE" if offset > ALIGN_DEADZONE else
                 "GAUCHE" if offset < -ALIGN_DEADZONE else "ALIGNE")

    car = np.array([w / 2.0, h - 1.0])
    target = np.array([aim_x, aim_y])
    ctrl = np.array([w / 2.0, (car[1] + target[1]) / 2.0])
    ts = np.linspace(0, 1, 16)[:, None]
    bez = (1 - ts) ** 2 * car + 2 * (1 - ts) * ts * ctrl + ts ** 2 * target
    ahead = center[idx:]
    traj = np.vstack([bez, ahead]) if len(ahead) else bez

    result.update({"status": "OK", "trajectory": traj.astype(np.int32),
                   "offset": float(offset), "direction": direction})
    return result


class LaneModel:
    def __init__(self, device=None):
        if torch is None:
            raise RuntimeError("PyTorch absent : pip install torch torchvision")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(_ensure_weights(), map_location=self.device).eval()
        print(f"[YOLOPv2] charge sur {self.device}")

    def reset(self):
        pass

    def _preprocess(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(rgb, (INF_W, INF_H), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(img).to(self.device).float().permute(2, 0, 1) / 255.0
        return t.unsqueeze(0)

    @staticmethod
    def _mask(t, w, h):
        """(1,C,h,w) -> masque uint8 plein cadre. C>=2 -> argmax, sinon seuil."""
        if t.dim() == 4 and t.shape[1] >= 2:
            m = t.argmax(1)
        elif t.dim() == 4:
            m = (t[:, 0] > 0.5).int()
        else:
            m = (t > 0.5).int()
        m = m.squeeze().float().cpu().numpy()
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        return (m > 0.5).astype(np.uint8) * 255

    def detect(self, bgr):
        h, w = bgr.shape[:2]
        result = {"status": "NO_LANE", "drivable": None, "lanes": None,
                  "center_line": None, "trajectory": None,
                  "offset": 0.0, "direction": "NONE"}

        with torch.no_grad():
            out = self.model(self._preprocess(bgr))
        # YOLOPv2 : (det, drivable_seg, lane_seg)
        seg, ll = out[1], out[2]
        drivable = self._mask(seg, w, h)
        lanes = self._mask(ll, w, h)
        return build_lane_result(lanes, drivable, w, h)


# ---------------------------------------------------------------------------
# Rendu
# ---------------------------------------------------------------------------
DRIVABLE_COLOR = (0, 180, 0)     # vert (zone roulable)
LANE_COLOR = (0, 0, 255)         # rouge (lignes de voie)
TRAJ_COLOR = (0, 255, 255)       # jaune (trajectoire)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_overlay(bgr, result):
    overlay = bgr.copy()
    h, w = bgr.shape[:2]
    status = result.get("status", "NO_LANE")

    if result.get("drivable") is not None:
        layer = np.zeros_like(overlay)
        layer[result["drivable"] > 0] = DRIVABLE_COLOR
        overlay = cv2.addWeighted(overlay, 1.0, layer, 0.35, 0)
    if result.get("lanes") is not None:
        # affine les lignes a l'affichage (le masque est epais a cause des
        # labels dilates a l'entrainement)
        disp = cv2.erode(result["lanes"], np.ones((3, 3), np.uint8), iterations=2)
        overlay[disp > 0] = LANE_COLOR

    if status == "OK":
        cv2.line(overlay, (w // 2, h - 1), (w // 2, int(h * 0.62)), (255, 255, 255), 1)
        cv2.polylines(overlay, [result["trajectory"]], False, TRAJ_COLOR, 4)

    bar = overlay[0:80, 0:w]
    overlay[0:80, 0:w] = cv2.addWeighted(bar, 0.4, np.zeros_like(bar), 0.6, 0)
    if status == "OK":
        d = result["direction"]
        label, col = {"GAUCHE": ("<<<  ALLER A GAUCHE", (0, 200, 255)),
                      "DROITE": ("ALLER A DROITE  >>>", (0, 200, 255)),
                      "ALIGNE": ("ALIGNE  OK", (0, 255, 0))}[d]
        cv2.putText(overlay, label, (15, 38), FONT, 0.95, col, 2, cv2.LINE_AA)
        cv2.putText(overlay, f"offset={result['offset']:+.2f}", (15, 70),
                    FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(overlay, "VOIE NON DETECTEE", (15, 50), FONT, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    return overlay


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        m = LaneModel()
        img = cv2.imread(sys.argv[1])
        cv2.imwrite("yolop_out.png", draw_overlay(img, m.detect(img)))
        print("ecrit: yolop_out.png")
    else:
        print("usage: python yolop_lane.py <image.png>")
