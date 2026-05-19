"""RGB camera capture attached to the ego vehicle, JPEG save."""

from __future__ import annotations
from pathlib import Path

import numpy as np
from PIL import Image
import carla

# Shared team camera POV parameters
# POV outside the cockpit : # semantic/instance !see transparent materials.
CAMERA_LOCATION = (0.30, 0.0, 1.50)
CAMERA_ROTATION_PITCH = -5.0


class CameraCapture:
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
        self._last_frame: np.ndarray | None = None

    def attach(self) -> "carla.Sensor":
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
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

        self._sensor = self.world.spawn_actor(
            bp, transform, attach_to=self.ego
        )  # Stick to the car
        self._sensor.listen(self._on_image)
        return self._sensor

    def _on_image(self, image: "carla.Image") -> None:
        raw = np.frombuffer(
            image.raw_data, dtype=np.uint8
        )  # (H*W*4)1D - BytesInSeries in uint8
        bgra = raw.reshape(
            (image.height, image.width, 4)
        )  # (H, W, 4) 3D - BGRA in uint8
        self._last_frame = bgra[..., [2, 1, 0]].copy()  # (H, W, 3) 3D - RGB in uint8

    def save_last_frame(self, path: Path) -> None:
        """JPEG (quality=90)."""
        if self._last_frame is None:
            raise RuntimeError(
                "No buffered image. Call after at least one world.tick()."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self._last_frame).save(str(path), quality=90)
