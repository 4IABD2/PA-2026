from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline


class DepthEstimator:
    """Depth Anything v2 small — monocular depth estimation.

    The model outputs relative disparity (close = high value).
    Calibration fits depth = scale / (raw + eps) + shift against CARLA GT.
    """

    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
        device: str = "cpu",
        max_depth_m: float = 100.0,
    ) -> None:
        self.device = device
        self.max_depth_m = max_depth_m
        self._pipe = pipeline(
            "depth-estimation",
            model=model_name,
            device=device,
        )
        self._scale: float | None = None
        self._shift: float | None = None

    def _raw_disparity(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        result = self._pipe(Image.fromarray(image))
        raw = np.array(result["depth"], dtype=np.float32)
        if raw.shape[:2] != (h, w):
            raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR)
        return raw

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth from an RGB image.

        Args:
            image: RGB (H, W, 3), dtype uint8.
        Returns:
            Depth map (H, W) float32, in metres. Clamped to [0, max_depth_m].
        """
        raw = self._raw_disparity(image)

        if self._scale is not None:
            eps = 1e-6
            depth = self._scale / (raw + eps) + self._shift
        else:
            depth = raw

        return np.clip(depth, 0.0, self.max_depth_m)

    def calibrate(self, gt_depths: list[np.ndarray], images: list[np.ndarray]) -> None:
        """Fit scale and shift: depth_metres = scale / (raw_disparity + eps) + shift."""
        all_inv = []
        all_gt = []
        eps = 1e-6
        for img, gt in zip(images, gt_depths):
            raw = self._raw_disparity(img)
            mask = (gt > 0) & (gt < self.max_depth_m) & (raw > eps)
            all_inv.append(1.0 / (raw[mask] + eps))
            all_gt.append(gt[mask])

        inv_flat = np.concatenate(all_inv)
        gt_flat = np.concatenate(all_gt)

        A = np.stack([inv_flat, np.ones_like(inv_flat)], axis=1)
        result = np.linalg.lstsq(A, gt_flat, rcond=None)
        self._scale, self._shift = float(result[0][0]), float(result[0][1])

        residuals = gt_flat - (self._scale * inv_flat + self._shift)
        rmse = float(np.sqrt(np.mean(residuals**2)))
        print(
            f"Calibrated: scale={self._scale:.4f}, shift={self._shift:.4f}, RMSE={rmse:.2f}m"
        )
