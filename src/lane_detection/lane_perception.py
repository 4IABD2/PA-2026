import os
import urllib.request

import cv2
import numpy as np
import torch

try:
    from src.lane_detection.lane_geometry import lane_geometry
except ImportError:
    from lane_geometry import (
        lane_geometry,
    )

WEIGHTS_URL = "https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt"
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "yolopv2.pt")
INF_W, INF_H = 640, 480  # model input size (multiple of 32, 4:3 ratio)

DRIVABLE_COLOR = (0, 180, 0)
LANE_COLOR = (0, 0, 255)
TRAJ_COLOR = (0, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# --- YOLOPv2 model: RGB -> masks (drivable area + lane lines) ----------------
def _ensure_weights():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    if not os.path.exists(WEIGHTS_PATH):
        print(f"[YOLOPv2] downloading weights -> {WEIGHTS_PATH} ...")
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
    return WEIGHTS_PATH


class LaneDetector:
    """Loads YOLOPv2 (TorchScript) and produces, from an RGB image, masks
    (lane lines + drivable area) then lane geometry."""

    def __init__(self, weights=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        with open(weights or _ensure_weights(), "rb") as f:
            self.model = torch.jit.load(f, map_location=self.device).eval()

    def _preprocess(self, rgb):
        img = cv2.resize(rgb, (INF_W, INF_H), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(img).to(self.device).float().permute(2, 0, 1) / 255.0
        return t.unsqueeze(0)

    @staticmethod
    def _mask(t, w, h):
        if t.dim() == 4 and t.shape[1] >= 2:
            m = t.argmax(1)
        elif t.dim() == 4:
            m = (t[:, 0] > 0.5).int()
        else:
            m = (t > 0.5).int()
        m = m.squeeze().float().cpu().numpy()
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        return (m > 0.5).astype(np.uint8) * 255

    def detect(self, rgb):
        h, w = rgb.shape[:2]
        with torch.no_grad():
            out = self.model(self._preprocess(rgb))  # (det, drivable_seg, lane_seg)
        drivable = self._mask(out[1], w, h)
        lanes = self._mask(out[2], w, h)
        return lane_geometry(lanes, drivable, w, h)


_DETECTOR = None


def estimate(rgb, detector=None):
    global _DETECTOR
    if detector is None:
        if _DETECTOR is None:
            _DETECTOR = LaneDetector()
        detector = _DETECTOR
    r = detector.detect(rgb)
    return r["direction"], r["angle"], r["offset"]


def draw_overlay(bgr, result):
    overlay = bgr.copy()
    h, w = bgr.shape[:2]
    status = result.get("status", "NO_LANE")

    if result.get("drivable") is not None:
        layer = np.zeros_like(overlay)
        layer[result["drivable"] > 0] = DRIVABLE_COLOR
        overlay = cv2.addWeighted(overlay, 1.0, layer, 0.35, 0)
    if result.get("lanes") is not None:
        disp = cv2.erode(result["lanes"], np.ones((3, 3), np.uint8), iterations=2)
        overlay[disp > 0] = LANE_COLOR

    if status == "OK":
        cv2.line(overlay, (w // 2, h - 1), (w // 2, int(h * 0.62)), (255, 255, 255), 1)
        cv2.polylines(overlay, [result["trajectory"]], False, TRAJ_COLOR, 4)

    bar = overlay[0:80, 0:w]
    overlay[0:80, 0:w] = cv2.addWeighted(bar, 0.4, np.zeros_like(bar), 0.6, 0)
    if status == "OK":
        d = result["direction"]
        label, col = {
            "GAUCHE": ("<<<  TURN LEFT", (0, 200, 255)),
            "DROITE": ("TURN RIGHT  >>>", (0, 200, 255)),
            "ALIGNE": ("ALIGNED  OK", (0, 255, 0)),
        }[d]
        cv2.putText(overlay, label, (15, 38), FONT, 0.95, col, 2, cv2.LINE_AA)
        cv2.putText(
            overlay,
            f"angle={result['angle']:+.1f} deg",
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
            "LANE NOT DETECTED",
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
        bgr = cv2.imread(sys.argv[1])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        det = LaneDetector()
        res = det.detect(rgb)
        print(
            "direction =", res["direction"], "| angle =", round(res["angle"], 1), "deg"
        )
        cv2.imwrite("perception_out.png", draw_overlay(bgr, res))
        print("written: perception_out.png")
    else:
        print("usage: python lane_perception.py <image.png>")
