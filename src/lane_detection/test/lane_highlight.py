"""Detection de voie (OpenCV, RGB pur -- aucune dependance CARLA).

But : donner un retour visuel simple ->
  1. le TRACE de la voie courante (lignes gauche/droite + remplissage),
  2. la TRAJECTOIRE que la voiture doit suivre pour rester alignee au centre,
  3. une consigne d'alignement (gauche / droite / aligne).

Approche GABARIT (template matching), plus robuste que l'ajustement de lignes :
on glisse une voie de forme fixe (deux bords convergeant vers un point de fuite)
sur differents centres et caps, et on garde celle qui se SUPERPOSE le mieux aux
marquages. La geometrie est donc toujours propre (jamais croisee, largeur stable)
et le score de recouvrement sert de CONFIANCE : un carrefour / une absence de voie
ne matche aucun gabarit -> score bas -> on ne trace rien.

Tous les reglages sont des constantes en haut de fichier.
"""

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 1. Masque des marquages (couleur + structures fines)
# ---------------------------------------------------------------------------
# Region d'interet : trapeze (fractions de l'image, x puis y), large en bas.
ROI_VERTICES = np.array(
    [
        [0.00, 1.00],
        [0.28, 0.55],
        [0.72, 0.55],
        [1.00, 1.00],
    ],
    dtype=np.float32,
)

WHITE_L_MIN = 155  # luminosite mini d'un marquage (bas -> marquages pales)
WHITE_S_MAX = 85  # saturation maxi (le blanc est peu sature)
YELLOW_LOW = (15, 60, 70)  # (H, L, S)
YELLOW_HIGH = (40, 230, 255)
TOPHAT_KERNEL = 17  # > largeur d'un marquage : garde les structures fines
TOPHAT_MIN = 18

MIN_AREA = 40  # px : en dessous = bruit
MAX_LINE_THICKNESS = 45  # px : epaisseur (aire/longueur) maxi d'un marquage
BLOB_FILL_RATIO = 0.65  # surface/bbox au-dela = bloc plein (voiture, panneau)
BLOB_MIN_SIDE = 30

_TOPHAT_K = cv2.getStructuringElement(cv2.MORPH_RECT, (TOPHAT_KERNEL, TOPHAT_KERNEL))


def _roi_mask(h, w):
    pts = (ROI_VERTICES * [w, h]).astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def _filter_shapes(mask):
    """Garde les composantes fines/allongees (marquages), jette bruit et blocs."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < MIN_AREA:
            continue
        short_side = min(bw, bh)
        fill_ratio = area / float(bw * bh)
        diag = max((bw * bw + bh * bh) ** 0.5, 1.0)
        est_thickness = area / diag  # aire/longueur : robuste aux diagonales
        if fill_ratio > BLOB_FILL_RATIO and short_side > BLOB_MIN_SIDE:
            continue
        if est_thickness > MAX_LINE_THICKNESS:
            continue
        keep[labels == i] = 255
    return keep


def lane_marking_mask(bgr):
    """Masque binaire (uint8) des marquages de voie."""
    h, w = bgr.shape[:2]
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]

    white = cv2.inRange(hls, (0, WHITE_L_MIN, 0), (179, 255, WHITE_S_MAX))
    yellow = cv2.inRange(hls, YELLOW_LOW, YELLOW_HIGH)

    tophat = cv2.morphologyEx(l_channel, cv2.MORPH_TOPHAT, _TOPHAT_K)
    thin_bright = cv2.inRange(tophat, TOPHAT_MIN, 255)

    mask = cv2.bitwise_or(cv2.bitwise_and(white, thin_bright), yellow)
    mask = cv2.bitwise_and(mask, _roi_mask(h, w))
    return _filter_shapes(mask)


# ---------------------------------------------------------------------------
# 2. Detecteur de voie par gabarit
# ---------------------------------------------------------------------------
VP_Y_FRAC = 0.45  # hauteur du point de fuite des gabarits
HALF_WIDTH_FRAC = 0.21  # demi-largeur de voie (en bas)
CENTER_MIN, CENTER_MAX = 0.38, 0.62  # balayage du centre de voie (frac w)
CENTER_STEPS = 17
VPX_MIN, VPX_MAX = 0.44, 0.56  # balayage du cap (x du point de fuite, resserre)
VPX_STEPS = 7
SAMPLES = 30  # points echantillonnes le long d'une ligne
TOL = 18  # tolerance laterale (px) pour qu'un point "matche"
DASH_BRIDGE = 41  # fermeture verticale (px) pour relier les pointilles
MATCH_MIN = 0.30  # score mini pour valider une voie (sinon : pas de voie)

ALIGN_DEADZONE = 0.06  # zone morte (offset normalise) -> "aligne"
LOOKAHEAD_FRAC = 0.55  # ou l'on vise pour la trajectoire (0=loin, 1=pres)


class LaneDetector:
    def __init__(self, image_width=800, image_height=600):
        self.w = int(image_width)
        self.h = int(image_height)
        self.vp_y = int(VP_Y_FRAC * self.h)
        self.hw = HALF_WIDTH_FRAC * self.w
        self.centers = np.linspace(
            CENTER_MIN * self.w, CENTER_MAX * self.w, CENTER_STEPS
        )
        self.vpxs = np.linspace(VPX_MIN * self.w, VPX_MAX * self.w, VPX_STEPS)
        self.ys = np.linspace(self.vp_y, self.h - 1, SAMPLES)
        self._t = (self.ys - (self.h - 1)) / (self.vp_y - (self.h - 1))  # 0 bas -> 1 VP
        self._yi = self.ys.astype(np.int32)
        self._hk = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * TOL + 1, 1))
        self._vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, DASH_BRIDGE))

    def reset(self):
        pass

    def _xs(self, x_bottom, vp_x):
        """x du gabarit a chaque y : ligne de (x_bottom, bas) vers (vp_x, VP)."""
        return x_bottom + self._t * (vp_x - x_bottom)

    def _score(self, scoremask, xs):
        xi = np.clip(xs, 0, self.w - 1).astype(np.int32)
        return float((scoremask[self._yi, xi] > 0).mean())

    def detect(self, bgr):
        mask = lane_marking_mask(bgr)
        # relie les pointilles (vertical) puis elargit lateralement (+-TOL)
        sm = cv2.dilate(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._vk), self._hk)

        result = {
            "mask": mask,
            "status": "NO_LANE",
            "score": 0.0,
            "left": None,
            "right": None,
            "center_line": None,
            "trajectory": None,
            "offset": 0.0,
            "direction": "NONE",
        }

        best = None
        for vpx in self.vpxs:
            for c in self.centers:
                ls = self._score(sm, self._xs(c - self.hw, vpx))
                rs = self._score(sm, self._xs(c + self.hw, vpx))
                score = (ls * rs) ** 0.5  # moyenne geometrique : exige les 2 bords
                if best is None or score > best[0]:
                    best = (score, c, vpx)

        result["score"] = round(best[0], 3)
        if best[0] < MATCH_MIN:
            return result  # aucune voie ne se superpose -> on ne trace rien

        _, c, vpx = best
        lxs = self._xs(c - self.hw, vpx)
        rxs = self._xs(c + self.hw, vpx)
        cxs = 0.5 * (lxs + rxs)
        left = np.stack([lxs, self.ys], axis=1).astype(np.int32)
        right = np.stack([rxs, self.ys], axis=1).astype(np.int32)
        center_line = np.stack([cxs, self.ys], axis=1).astype(np.int32)

        # consigne d'alignement : centre de voie vise vs centre camera
        idx = int(np.clip(round(LOOKAHEAD_FRAC * (SAMPLES - 1)), 0, SAMPLES - 1))
        aim_x, aim_y = cxs[idx], self.ys[idx]
        offset = (aim_x - self.w / 2.0) / (self.w / 2.0)
        direction = (
            "DROITE"
            if offset > ALIGN_DEADZONE
            else "GAUCHE" if offset < -ALIGN_DEADZONE else "ALIGNE"
        )

        result.update(
            {
                "status": "OK",
                "left": left,
                "right": right,
                "center_line": center_line,
                "trajectory": self._trajectory(cxs, aim_x, aim_y),
                "offset": float(offset),
                "direction": direction,
            }
        )
        return result

    def _trajectory(self, cxs, aim_x, aim_y):
        """Chemin que la voiture doit suivre : part de sa position actuelle
        (bas, centre camera) et rejoint en douceur le centre de la voie devant,
        puis suit le centre de voie. Bezier quadratique pour la jonction."""
        car = np.array([self.w / 2.0, self.h - 1.0])
        target = np.array([aim_x, aim_y])
        ctrl = np.array(
            [self.w / 2.0, (car[1] + target[1]) / 2.0]
        )  # garde le cap puis tourne
        ts = np.linspace(0.0, 1.0, 16)[:, None]
        bez = (1 - ts) ** 2 * car + 2 * (1 - ts) * ts * ctrl + ts**2 * target
        # prolonge le long du centre de voie au-dela du point vise
        idx = int(np.clip(round(LOOKAHEAD_FRAC * (SAMPLES - 1)), 0, SAMPLES - 1))
        ahead = np.stack([cxs[:idx], self.ys[:idx]], axis=1)
        return np.vstack([bez, ahead]).astype(np.int32)


# ---------------------------------------------------------------------------
# 3. Rendu
# ---------------------------------------------------------------------------
LEFT_COLOR = (0, 0, 255)  # rouge (BGR)
RIGHT_COLOR = (255, 0, 0)  # bleu
FILL_COLOR = (0, 180, 0)  # vert
TRAJ_COLOR = (0, 255, 255)  # jaune
NEAR_FRAC = 0.66  # on ne dessine la voie que dans le champ proche
# (bande large pres de la voiture, pas un pic)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_overlay(bgr, result):
    """Dessine la voie (bords + remplissage), la trajectoire et la consigne."""
    overlay = bgr.copy()
    h, w = bgr.shape[:2]
    status = result.get("status", "NO_LANE")

    if status == "OK":
        left, right = result["left"], result["right"]
        # voie dessinee seulement dans le champ proche -> bande large (pas un pic)
        ny = int(NEAR_FRAC * h)
        ln = left[left[:, 1] >= ny]
        rn = right[right[:, 1] >= ny]
        if len(ln) >= 2 and len(rn) >= 2:
            area = np.vstack([ln, rn[::-1]])
            fill = overlay.copy()
            cv2.fillPoly(fill, [area], FILL_COLOR)
            overlay = cv2.addWeighted(overlay, 0.75, fill, 0.25, 0)
            cv2.polylines(overlay, [ln], False, LEFT_COLOR, 6)
            cv2.polylines(overlay, [rn], False, RIGHT_COLOR, 6)
        # trajectoire d'alignement (jaune, epaisse) + repere voiture (blanc)
        cv2.line(overlay, (w // 2, h - 1), (w // 2, int(h * 0.62)), (255, 255, 255), 1)
        cv2.polylines(overlay, [result["trajectory"]], False, TRAJ_COLOR, 4)

    # bandeau HUD
    bar = overlay[0:80, 0:w]
    overlay[0:80, 0:w] = cv2.addWeighted(bar, 0.4, np.zeros_like(bar), 0.6, 0)
    if status == "OK":
        d = result["direction"]
        label, col = {
            "GAUCHE": ("<<<  ALLER A GAUCHE", (0, 200, 255)),
            "DROITE": ("ALLER A DROITE  >>>", (0, 200, 255)),
            "ALIGNE": ("ALIGNE  OK", (0, 255, 0)),
        }[d]
        cv2.putText(overlay, label, (15, 38), FONT, 0.95, col, 2, cv2.LINE_AA)
        cv2.putText(
            overlay,
            f"offset={result['offset']:+.2f}  score={result['score']:.2f}",
            (15, 70),
            FONT,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            overlay,
            "VOIE NON DETECTEE",
            (15, 50),
            FONT,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return overlay


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        det = LaneDetector(img.shape[1], img.shape[0])
        cv2.imwrite("lane_out.png", draw_overlay(img, det.detect(img)))
        print("ecrit: lane_out.png")
    else:
        print("usage: python lane_highlight.py <image.png>")
