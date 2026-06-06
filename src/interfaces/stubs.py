"""Ground-truth CARLA stubs — replaced by real perception models once ready.

Never use in production: stubs read CARLA internals, bypassing real CV.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.interfaces.perception_types import (
    DetectedObject,
    LanesInfo,
    ObjectClass,
)

if TYPE_CHECKING:
    import carla  # noqa: F401


# replaced by: src/perception/detection/ — Franck's YOLO-based detector
class CarlaGTObjectDetector:
    def __init__(self, world: "carla.World", ego_vehicle: "carla.Actor") -> None:
        self.world = world
        self.ego = ego_vehicle

    def detect(self, image: np.ndarray) -> list[DetectedObject]:
        raise NotImplementedError("TODO: project actor bboxes via CARLA camera matrix")


# replaced by: src/perception/depth/ — Franck's MiDaS / Depth-Anything estimator
class CarlaGTDepthEstimator:
    # CARLA depth encoding: (R + G*256 + B*65536) / 16_777_215 * 1000 m  (raw_data is BGRA)
    def __init__(self, depth_sensor: "carla.Sensor") -> None:
        self.depth_sensor = depth_sensor
        self._last_depth: np.ndarray | None = None
        depth_sensor.listen(self._on_depth)

    def _on_depth(self, raw_image) -> None:
        arr = np.frombuffer(raw_image.raw_data, dtype=np.uint8).reshape(
            raw_image.height, raw_image.width, 4
        )
        r = arr[:, :, 2].astype(np.float32)
        g = arr[:, :, 1].astype(np.float32)
        b = arr[:, :, 0].astype(np.float32)
        self._last_depth = (r + g * 256.0 + b * 65536.0) / 16_777_215.0 * 1000.0

    def estimate(self, image: np.ndarray) -> np.ndarray:
        if self._last_depth is None:
            raise RuntimeError("No depth frame received yet — is the sensor running?")
        return self._last_depth


# replaced by: src/perception/lanes/ — Karim's lane-detection model
class CarlaGTLaneDetector:
    def __init__(self, world: "carla.World", ego_vehicle: "carla.Actor") -> None:
        self.world = world
        self.ego = ego_vehicle

    def detect(self, _image: np.ndarray) -> LanesInfo:
        import math

        v_loc = self.ego.get_transform().location
        wp = self.world.get_map().get_waypoint(v_loc, project_to_road=True)

        yaw_rad = math.radians(wp.transform.rotation.yaw)
        fwd_x = math.cos(yaw_rad)
        fwd_y = math.sin(yaw_rad)

        dx = v_loc.x - wp.transform.location.x
        dy = v_loc.y - wp.transform.location.y

        # cross product z-component: positive = vehicle is right of waypoint heading
        lateral = fwd_x * dy - fwd_y * dx
        half_width = (wp.lane_width or 3.5) / 2.0
        center_offset = max(-1.0, min(1.0, lateral / half_width))

        return LanesInfo(left_line=None, right_line=None, center_offset=center_offset)


__all__ = [
    "CarlaGTObjectDetector",
    "CarlaGTDepthEstimator",
    "CarlaGTLaneDetector",
]
