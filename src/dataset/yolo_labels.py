"""YOLO label generation (normalized 2D bboxes) for the dataset."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import carla  # noqa: F401

# Class mapping documented in the root README.
YOLO_CLASS_MAPPING: dict[str, int] = {
    "vehicle": 0,
    "walker": 1,
    "traffic_light": 2,
}

# Type of a label row: (class_id, x_center, y_center, width, height) normalized [0, 1]
YoloLabel = tuple[int, float, float, float, float]


class YoloLabeler:
    """Compute YOLO labels from CARLA actors visible in the camera frame."""

    def __init__(
        self,
        world: "carla.World",
        ego: "carla.Vehicle",
        camera_sensor: "carla.Sensor",
        image_w: int = 1280,
        image_h: int = 720,
        max_distance_m: float = 80.0,
    ) -> None:
        self.world = world
        self.ego = ego
        self.camera_sensor = camera_sensor
        self.image_w = image_w
        self.image_h = image_h
        self.max_distance_m = max_distance_m

    def compute_labels(self) -> list[YoloLabel]:
        """Return YOLO labels for the current frame.

        TODO (Franck): implement 3D->2D projection of visible actors using
        the CARLA intrinsic matrix (cf. carla.Sensor.calibration and
        world_to_camera_matrix). Filter:
        - distance ego->actor > max_distance_m
        - actor outside camera frustum
        - occluded actor (optional, can be skipped for the skeleton)
        """
        raise NotImplementedError(
            "3D->2D projection not implemented yet. "
            "To be completed by Franck in a later session. "
            "See docstring for the spec."
        )

    def save(self, path: Path) -> None:
        """Compute and write labels in YOLO format."""
        labels = self.compute_labels()
        self.write_labels_to_file(labels, path)

    @staticmethod
    def write_labels_to_file(labels: list[YoloLabel], path: Path) -> None:
        """Write a list of (already computed) labels in standard YOLO format.

        Static method so it is testable without CARLA.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
            for (cls_id, x, y, w, h) in labels
        ]
        path.write_text("\n".join(lines) + ("\n" if lines else ""))
