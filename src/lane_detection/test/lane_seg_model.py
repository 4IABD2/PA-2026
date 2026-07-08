"""Phase 3 : inference avec le modele fine-tune sur CARLA (weights/lane_seg.pt).

Meme interface et meme sortie que yolop_lane.LaneModel -> compatible avec
draw_overlay et live_highlight. Pour basculer le viewer sur le modele CARLA :
remplacer dans live_highlight.py
    from yolop_lane import LaneModel, draw_overlay
par
    from lane_seg_model import LaneModel
    from yolop_lane import draw_overlay
"""

import cv2
import numpy as np
import torch

from train_lane_seg import build_model, TRAIN_W, TRAIN_H, MEAN, STD, WEIGHTS_OUT
from yolop_lane import build_lane_result, draw_overlay  # noqa: F401 (reexport)


class LaneModel:
    def __init__(self, weights=WEIGHTS_OUT, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model()
        self.model.load_state_dict(torch.load(weights, map_location=self.device))
        self.model.to(self.device).eval()
        self._mean = MEAN.to(self.device)
        self._std = STD.to(self.device)
        print(f"[lane_seg] charge {weights} sur {self.device}")

    def reset(self):
        pass

    def _preprocess(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(rgb, (TRAIN_W, TRAIN_H), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(img).to(self.device).float().permute(2, 0, 1) / 255.0
        x = (x - self._mean) / self._std
        return x.unsqueeze(0)

    def detect(self, bgr):
        h, w = bgr.shape[:2]
        with torch.no_grad():
            out = self.model(self._preprocess(bgr))["out"]  # 1x3xH'xW'
        pred = out.argmax(1).squeeze().cpu().numpy().astype(np.uint8)
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
        lanes = (pred == 1).astype(np.uint8) * 255
        drivable = (pred == 2).astype(np.uint8) * 255
        return build_lane_result(lanes, drivable, w, h)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        m = LaneModel()
        img = cv2.imread(sys.argv[1])
        cv2.imwrite("laneseg_out.png", draw_overlay(img, m.detect(img)))
        print("ecrit: laneseg_out.png")
    else:
        print("usage: python lane_seg_model.py <image.png>  (apres entrainement)")
