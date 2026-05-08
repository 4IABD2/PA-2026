"""Ground-truth depth capture from CARLA, decoded to meters and saved as .npy."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.dataset.camera_capture import CAMERA_LOCATION, CAMERA_ROTATION_PITCH

if TYPE_CHECKING:
    import carla  # noqa: F401


def decode_carla_depth(
    rgb: np.ndarray,
    max_depth_m: float = 1000.0,
) -> np.ndarray:
    """Decode CARLA's 3-channel depth encoding into float32 meters, clipped to ``max_depth_m``.

    Formula: ``meters = ((R + G*256 + B*256**2) / (256**3 - 1)) * 1000``.
    """
    rgb_f = rgb.astype(np.float32)
    normalized = (
        rgb_f[..., 0] + rgb_f[..., 1] * 256.0 + rgb_f[..., 2] * (256.0 * 256.0)
    ) / (256.0**3 - 1.0)
    meters = normalized * 1000.0
    return np.clip(meters, 0.0, max_depth_m).astype(np.float32)


class DepthCapture:
    """Wrapper for sensor.camera.depth with decoding to meters."""

    def __init__(
        self,
        world: "carla.World",
        ego: "carla.Vehicle",
        width: int = 1280,
        height: int = 720,
        fov: int = 90,
        max_depth_m: float = 100.0,
    ) -> None:
        self.world = world
        self.ego = ego
        self.width = width
        self.height = height
        self.fov = fov
        self.max_depth_m = max_depth_m
        self._sensor: "carla.Sensor | None" = None
        self._last_rgb: np.ndarray | None = None

    def attach(self) -> "carla.Sensor":
        import carla

        bp = self.world.get_blueprint_library().find("sensor.camera.depth")
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
        raw = np.frombuffer(image.raw_data, dtype=np.uint8)
        bgra = raw.reshape((image.height, image.width, 4))
        self._last_rgb = bgra[..., [2, 1, 0]].copy()

    def save_last_frame(self, path: Path) -> None:
        if self._last_rgb is None:
            raise RuntimeError(
                "No buffered image. Call after at least one world.tick()."
            )
        depth_m = decode_carla_depth(self._last_rgb, max_depth_m=self.max_depth_m)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, depth_m)

    def destroy(self) -> None:
        if self._sensor is not None:
            self._sensor.stop()
            self._sensor.destroy()
            self._sensor = None
