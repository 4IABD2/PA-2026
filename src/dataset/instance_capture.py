"""CARLA instance segmentation capture and per-instance colorization."""

from __future__ import annotations

import colorsys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from src.dataset.camera_capture import CAMERA_LOCATION, CAMERA_ROTATION_PITCH

if TYPE_CHECKING:
    import carla  # noqa: F401


def pack_instance_carla(rgb: np.ndarray) -> np.ndarray:
    """Pack a raw CARLA instance segmentation frame into a single uint32 array.

    CARLA encodes:
      R = semantic class ID (0-28, CARLA 0.9.13+ mapping)
      G = instance_id high byte
      B = instance_id low byte

    Packed layout (uint32):
      bits [23:16] = class_id
      bits [15: 8] = G
      bits [ 7: 0] = B

    Unpack:
      class_id    = (packed >> 16) & 0xFF
      instance_id = packed & 0xFFFF

    Args:
        rgb: (H, W, 3) uint8 array (R, G, B).

    Returns:
        (H, W) uint32 packed array.
    """
    r = rgb[..., 0].astype(np.uint32)
    g = rgb[..., 1].astype(np.uint32)
    b = rgb[..., 2].astype(np.uint32)
    return (r << 16) | (g << 8) | b


# Golden ratio conjugate — used to spread instance hues evenly.
_GOLDEN_RATIO_CONJUGATE = 0.6180339887


def colorize_instance(packed: np.ndarray) -> np.ndarray:
    """Render a packed instance map as RGB with deterministic per-instance colors.

    Each unique instance_id maps to (h, s, v) = ((iid * φ) mod 1, 0.6, 0.95)
    where φ is the golden ratio conjugate — gives well-spread, non-clashing
    hues. instance_id == 0 (background / unlabeled) → black.

    A given instance keeps the same color across frames of the same run, so
    the visualization is useful for tracking objects through time.

    Args:
        packed: (H, W) uint32 array as returned by pack_instance_carla.

    Returns:
        (H, W, 3) uint8 RGB image.
    """
    instance_ids = (packed & 0xFFFF).astype(np.uint32)
    unique_ids = np.unique(instance_ids)

    rgb_lookup = np.zeros((len(unique_ids), 3), dtype=np.uint8)
    for i, iid in enumerate(unique_ids):
        if iid == 0:
            continue  # already (0, 0, 0)
        hue = (float(iid) * _GOLDEN_RATIO_CONJUGATE) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.6, 0.95)
        rgb_lookup[i] = (int(r * 255), int(g * 255), int(b * 255))

    # np.unique returns sorted values, so searchsorted gives the index per pixel.
    sort_idx = np.searchsorted(unique_ids, instance_ids)
    return rgb_lookup[sort_idx]


class InstanceCapture:
    """Wrapper for sensor.camera.instance_segmentation. Same lifecycle as DepthCapture."""

    def __init__(
        self,
        world: "carla.World",
        ego: "carla.Vehicle",
        width: int = 1280,
        height: int = 720,
        fov: int = 90,
    ) -> None:
        self.world = world
        self.ego = ego
        self.width = width
        self.height = height
        self.fov = fov
        self._sensor: "carla.Sensor | None" = None
        self._last_rgb: np.ndarray | None = None

    def attach(self) -> "carla.Sensor":
        import carla

        bp = self.world.get_blueprint_library().find(
            "sensor.camera.instance_segmentation"
        )
        bp.set_attribute("image_size_x", str(self.width))
        bp.set_attribute("image_size_y", str(self.height))
        bp.set_attribute("fov", str(self.fov))

        transform = carla.Transform(
            carla.Location(
                x=CAMERA_LOCATION[0],
                y=CAMERA_LOCATION[1],
                z=CAMERA_LOCATION[2],
            ),
            carla.Rotation(pitch=CAMERA_ROTATION_PITCH),
        )

        self._sensor = self.world.spawn_actor(bp, transform, attach_to=self.ego)
        self._sensor.listen(self._on_image)
        return self._sensor

    def _on_image(self, image: "carla.Image") -> None:
        """Copy raw pixels to numpy immediately (BGRA -> RGB)."""
        raw = np.frombuffer(image.raw_data, dtype=np.uint8)
        bgra = raw.reshape((image.height, image.width, 4))
        self._last_rgb = bgra[..., [2, 1, 0]].copy()

    def save_last_frame(self, npy_path: Path, viz_path: Path) -> None:
        """Pack the last frame to uint32 and save both .npy + .png viz."""
        if self._last_rgb is None:
            raise RuntimeError(
                "No buffered image. Call after at least one world.tick()."
            )
        packed = pack_instance_carla(self._last_rgb)
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, packed)

        viz = colorize_instance(packed)
        viz_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(viz).save(str(viz_path))

    def destroy(self) -> None:
        if self._sensor is not None:
            self._sensor.stop()
            self._sensor.destroy()
            self._sensor = None
