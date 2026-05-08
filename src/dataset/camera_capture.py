"""RGB camera capture attached to the ego vehicle, JPEG save."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    import carla  # noqa: F401

# Shared team camera POV (cf. root README "CARLA conventions")
CAMERA_LOCATION = (0.5, -0.3, 1.2)
CAMERA_ROTATION_PITCH = -5.0


class CameraCapture:
    """Wrapper for sensor.camera.rgb with a callback that buffers the last image."""

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
        """Spawn and attach the sensor to the ego vehicle, return the sensor."""
        import carla

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

        self._sensor = self.world.spawn_actor(bp, transform, attach_to=self.ego)
        self._sensor.listen(self._on_image)
        return self._sensor

    def _on_image(self, image: "carla.Image") -> None:
        """CARLA callback — copy pixels to numpy immediately."""
        raw = np.frombuffer(image.raw_data, dtype=np.uint8)
        bgra = raw.reshape((image.height, image.width, 4))
        self._last_frame = bgra[..., [2, 1, 0]].copy()  # BGRA -> RGB

    def save_last_frame(self, path: Path) -> None:
        """Write the last buffered image as JPEG quality 90."""
        if self._last_frame is None:
            raise RuntimeError(
                "No buffered image. Call after at least one world.tick()."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self._last_frame).save(str(path), quality=90)

    def destroy(self) -> None:
        if self._sensor is not None:
            self._sensor.stop()
            self._sensor.destroy()
            self._sensor = None
