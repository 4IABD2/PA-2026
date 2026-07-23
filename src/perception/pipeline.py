from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.interfaces.perception_types import DetectedObject
from src.perception.depth.estimator import DepthEstimator
from src.perception.yolo.detector import YoloDetector

_DEFAULT_YOLO_WEIGHTS = "src/perception/yolo/weights/best.pt"
_DEFAULT_DEPTH_CALIB = "src/perception/depth/calibration.json"


def _auto_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _distance_for_bbox(
    depth_map: np.ndarray, bbox: tuple[int, int, int, int], max_depth_m: float
) -> float | None:
    """Distance d'un objet = médiane de la depth sur la zone centrale du bbox.

    On rétrécit au centre (50%) pour éviter le fond qui déborde du bbox, et on
    ignore les valeurs invalides (0 ou clampées au max). Médiane = robuste aux
    pixels aberrants. None si aucune depth valide.
    """
    x1, y1, x2, y2 = bbox
    h, w = depth_map.shape[:2]
    mx, my = (x2 - x1) // 4, (y2 - y1) // 4
    cx1, cy1 = max(0, x1 + mx), max(0, y1 + my)
    cx2, cy2 = min(w, x2 - mx), min(h, y2 - my)
    if cx2 <= cx1 or cy2 <= cy1:
        cx1, cy1, cx2, cy2 = x1, y1, x2, y2

    patch = depth_map[cy1:cy2, cx1:cx2]
    valid = patch[(patch > 0) & (patch < max_depth_m)]
    if valid.size == 0:
        return None
    return round(float(np.median(valid)), 2)


class PerceptionPipeline:
    """Charge YOLO + Depth une fois, puis fusionne par image."""

    def __init__(
        self,
        yolo_weights: str = _DEFAULT_YOLO_WEIGHTS,
        depth_calibration: str | None = _DEFAULT_DEPTH_CALIB,
        depth_model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
        device: str | None = None,
        max_depth_m: float = 100.0,
        require_calibration: bool = False,
    ) -> None:
        self.device = device or _auto_device()
        calib_exists = bool(depth_calibration) and Path(depth_calibration).exists()
        if not calib_exists and require_calibration:
            raise RuntimeError(
                f"Depth calibration file not found: {depth_calibration!r}. "
                "Training with uncalibrated depth produces meaningless distance "
                "observations for the entire run -- fix the path or run "
                "src/perception/depth/calibrate.py first."
            )
        if not calib_exists:
            print(
                f"WARNING: depth calibration file not found ({depth_calibration!r}) -- "
                "falling back to UNCALIBRATED raw disparity. Distance-based "
                "observations/rewards will be meaningless until this is fixed."
            )
        calib = depth_calibration if calib_exists else None
        self.detector = YoloDetector(weights_path=yolo_weights, device=self.device)
        self.depth = DepthEstimator(
            model_name=depth_model_name,
            device=self.device,
            max_depth_m=max_depth_m,
            calibration_path=calib,
        )

    def perceive(
        self, image_rgb: np.ndarray
    ) -> tuple[list[DetectedObject], np.ndarray]:
        """Image RGB (H, W, 3) uint8 → (objets avec distance_m, depth map mètres)."""
        depth_map = self.depth.estimate(image_rgb)
        objects = self.detector.detect(image_rgb)
        for obj in objects:
            obj.distance_m = _distance_for_bbox(
                depth_map, obj.bbox, self.depth.max_depth_m
            )
        return objects, depth_map

    def perceive_dict(self, image_rgb: np.ndarray) -> list[dict]:
        """Variante simple : liste de dicts JSON-sérialisables (pour une API)."""
        objects, _ = self.perceive(image_rgb)
        return [
            {
                "class": o.class_name.value,
                "bbox": list(o.bbox),
                "confidence": round(o.confidence, 3),
                "distance_m": o.distance_m,
            }
            for o in objects
        ]


def main() -> None:
    import cv2

    parser = argparse.ArgumentParser(description="Test de la pipeline de perception.")
    parser.add_argument("--image", type=Path, required=True, help="Chemin image (.jpg)")
    parser.add_argument("--yolo-weights", default=_DEFAULT_YOLO_WEIGHTS)
    parser.add_argument("--calibration", default=_DEFAULT_DEPTH_CALIB)
    parser.add_argument("--device", default=None, help="cuda / cpu (defaut: auto)")
    args = parser.parse_args()

    bgr = cv2.imread(str(args.image))
    if bgr is None:
        raise SystemExit(f"Image illisible : {args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    perc = PerceptionPipeline(
        yolo_weights=args.yolo_weights,
        depth_calibration=args.calibration,
        device=args.device,
    )
    detections = perc.perceive_dict(rgb)
    print(json.dumps(detections, indent=2, ensure_ascii=False))
    print(f"\n{len(detections)} objet(s) détecté(s).")


if __name__ == "__main__":
    main()
