"""Génère une grille d'images avec détections YOLO + profondeur estimée.

Usage:
    uv run -m src.perception.visualize_batch \
        --images data/runs/2026-05-20_town01_clearnoon/images \
        --yolo-weights src/perception/yolo/weights/best.pt \
        --depth-gt data/runs/2026-05-20_town01_clearnoon/depth \
        --output batch_demo.jpg \
        --n 9 --every 30
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from src.perception.depth.estimator import DepthEstimator
from src.perception.yolo.detector import YoloDetector

_COLORS: dict[str, tuple[int, int, int]] = {
    "vehicle": (0, 255, 0),
    "walker": (255, 255, 0),
    "red_light": (0, 0, 255),
    "yellow_light": (0, 200, 255),
    "green_light": (0, 255, 128),
}
_DEFAULT_COLOR = (255, 128, 0)


def _draw_detections(
    image: np.ndarray,
    detections: list,
    depth_map: np.ndarray | None,
) -> np.ndarray:
    img = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        label = det.class_name.value
        color = _COLORS.get(label, _DEFAULT_COLOR)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {det.confidence:.0%}"
        if depth_map is not None:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cy = min(cy, depth_map.shape[0] - 1)
            cx = min(cx, depth_map.shape[1] - 1)
            dist = depth_map[cy, cx]
            text += f" {dist:.0f}m"

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(
            img, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
        )

    return img


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Images directory")
    parser.add_argument("--yolo-weights", default="src/perception/yolo/weights/best.pt")
    parser.add_argument(
        "--depth-gt", default=None, help="Depth GT dir (for calibration)"
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", default="batch_demo.jpg")
    parser.add_argument("--n", type=int, default=9, help="Number of images in the grid")
    parser.add_argument("--every", type=int, default=30, help="Take every Nth image")
    args = parser.parse_args()

    images_dir = Path(args.images)
    all_paths = sorted(images_dir.glob("*.jpg"))
    selected = all_paths[:: args.every][: args.n]

    print(f"Loading YOLO from {args.yolo_weights}")
    detector = YoloDetector(weights_path=args.yolo_weights, device=args.device)

    print("Loading Depth Anything v2")
    depth_est = DepthEstimator(device=args.device)

    if args.depth_gt:
        gt_dir = Path(args.depth_gt)
        cal_images, cal_gts = [], []
        for p in selected[:5]:
            gt_path = gt_dir / f"{p.stem}.npy"
            if gt_path.exists():
                cal_images.append(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))
                cal_gts.append(np.load(str(gt_path)))
        if cal_images:
            print(f"Calibrating depth on {len(cal_images)} frames...")
            depth_est.calibrate(cal_gts, cal_images)

    det_panels: list[np.ndarray] = []
    depth_panels: list[np.ndarray] = []
    for path in selected:
        print(f"  {path.name}")
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        detections = detector.detect(rgb)
        depth_map = depth_est.estimate(rgb)

        det_panels.append(_draw_detections(rgb, detections, depth_map))

        norm = np.clip(depth_map / depth_est.max_depth_m, 0, 1)
        colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        depth_panels.append(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))

    def _make_grid(panels: list[np.ndarray]) -> np.ndarray:
        cols = math.ceil(math.sqrt(len(panels)))
        rows = math.ceil(len(panels) / cols)
        h, w = panels[0].shape[:2]
        th, tw = h // 2, w // 2
        grid = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
        for i, panel in enumerate(panels):
            r, c = divmod(i, cols)
            grid[r * th : (r + 1) * th, c * tw : (c + 1) * tw] = cv2.resize(
                panel, (tw, th)
            )
        return grid

    det_grid = _make_grid(det_panels)
    depth_grid = _make_grid(depth_panels)

    output = Path(args.output)
    det_path = output.with_stem(output.stem + "_detections")
    depth_path = output.with_stem(output.stem + "_depth")

    cv2.imwrite(str(det_path), cv2.cvtColor(det_grid, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(depth_path), cv2.cvtColor(depth_grid, cv2.COLOR_RGB2BGR))
    print(f"Saved to {det_path} and {depth_path}")


if __name__ == "__main__":
    main()
