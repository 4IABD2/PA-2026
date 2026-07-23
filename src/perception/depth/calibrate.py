from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.perception.depth.estimator import DepthEstimator

_DEFAULT_OUT = "src/perception/depth/calibration.json"


def _auto_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_frames(run: Path, n: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Charge n frames régulièrement espacées : (images RGB, depths GT mètres)."""
    images_dir = run / "images"
    depth_dir = run / "depth"
    if not images_dir.is_dir() or not depth_dir.is_dir():
        raise SystemExit(f"images/ ou depth/ manquant dans {run}")

    img_paths = sorted(images_dir.glob("*.jpg"))
    pairs = []
    for img_path in img_paths:
        depth_path = depth_dir / f"{img_path.stem}.npy"
        if depth_path.exists():
            pairs.append((img_path, depth_path))
    if not pairs:
        raise SystemExit(f"Aucune paire image/depth dans {run}")

    step = max(1, len(pairs) // n)
    pairs = pairs[::step][:n]

    images, depths = [], []
    for img_path, depth_path in pairs:
        bgr = cv2.imread(str(img_path))
        images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        depths.append(np.load(depth_path).astype(np.float32))
    return images, depths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibre le DepthEstimator (CARLA GT)."
    )
    parser.add_argument("--run", type=Path, required=True, help="Dossier d'une run")
    parser.add_argument("--n", type=int, default=15, help="Nb de frames (defaut: 15)")
    parser.add_argument(
        "--device", default=None, help="cuda / cpu (defaut: auto-détection)"
    )
    parser.add_argument("--out", type=Path, default=Path(_DEFAULT_OUT))
    args = parser.parse_args()

    device = args.device or _auto_device()
    print(f"Device: {device}")

    images, depths = _load_frames(args.run, args.n)
    print(f"{len(images)} frames chargées depuis {args.run}")

    estimator = DepthEstimator(device=device)
    estimator.calibrate(depths, images)
    estimator.save_calibration(args.out)
    print(f"Calibration sauvée : {args.out}")


if __name__ == "__main__":
    main()
