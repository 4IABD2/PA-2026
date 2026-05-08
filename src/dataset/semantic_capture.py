"""CARLA semantic segmentation capture and palette colorization."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from src.dataset.camera_capture import CAMERA_LOCATION, CAMERA_ROTATION_PITCH

if TYPE_CHECKING:
    import carla  # noqa: F401


def decode_semantic_carla(rgb: np.ndarray) -> np.ndarray:
    """Extract class IDs from a raw CARLA semantic segmentation frame.

    CARLA encodes the semantic class ID in the R channel (G=0, B=0).

    Args:
        rgb: (H, W, 3) uint8 array (R, G, B) as returned by the sensor.

    Returns:
        (H, W) uint8 array of class IDs in [0, 28] (CARLA 0.9.13+ mapping).
    """
    return rgb[..., 0].copy()


# CARLA 0.9.16 CityScape palette — official mapping for sensor.camera.semantic_segmentation.
# Class set was reorganized in 0.9.13 (29 classes, indices 0-28).
# See https://carla.readthedocs.io/en/0.9.16/ref_sensors/#semantic-segmentation-camera
CITYSCAPE_PALETTE = np.array(
    [
        [0, 0, 0],  # 0  Unlabeled
        [128, 64, 128],  # 1  Roads
        [244, 35, 232],  # 2  SideWalks
        [70, 70, 70],  # 3  Building
        [102, 102, 156],  # 4  Wall
        [190, 153, 153],  # 5  Fence
        [153, 153, 153],  # 6  Pole
        [250, 170, 30],  # 7  TrafficLight
        [220, 220, 0],  # 8  TrafficSign
        [107, 142, 35],  # 9  Vegetation
        [152, 251, 152],  # 10 Terrain
        [70, 130, 180],  # 11 Sky
        [220, 20, 60],  # 12 Pedestrian
        [255, 0, 0],  # 13 Rider
        [0, 0, 142],  # 14 Car
        [0, 0, 70],  # 15 Truck
        [0, 60, 100],  # 16 Bus
        [0, 80, 100],  # 17 Train
        [0, 0, 230],  # 18 Motorcycle
        [119, 11, 32],  # 19 Bicycle
        [110, 190, 160],  # 20 Static
        [170, 120, 50],  # 21 Dynamic
        [55, 90, 80],  # 22 Other
        [45, 60, 150],  # 23 Water
        [157, 234, 50],  # 24 RoadLine
        [81, 0, 81],  # 25 Ground
        [150, 100, 100],  # 26 Bridge
        [230, 150, 140],  # 27 RailTrack
        [180, 165, 180],  # 28 GuardRail
    ],
    dtype=np.uint8,
)


def colorize_semantic(class_ids: np.ndarray) -> np.ndarray:
    """Apply the CityScape palette to a class-ID map.

    Args:
        class_ids: (H, W) uint8 array, values in [0, 28].

    Returns:
        (H, W, 3) uint8 RGB image. Out-of-range IDs are clipped to the
        last palette entry (defensive).
    """
    safe = np.clip(class_ids, 0, len(CITYSCAPE_PALETTE) - 1)
    return CITYSCAPE_PALETTE[safe]


class SemanticCapture:
    """Wrapper for sensor.camera.semantic_segmentation. Same lifecycle as DepthCapture."""

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
            "sensor.camera.semantic_segmentation"
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
        """Decode the last frame and save both .npy (uint8 class IDs) and .png (palette viz)."""
        if self._last_rgb is None:
            raise RuntimeError(
                "No buffered image. Call after at least one world.tick()."
            )
        class_ids = decode_semantic_carla(self._last_rgb)
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, class_ids)

        viz = colorize_semantic(class_ids)
        viz_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(viz).save(str(viz_path))

    def destroy(self) -> None:
        if self._sensor is not None:
            self._sensor.stop()
            self._sensor.destroy()
            self._sensor = None
