"""Génération des labels YOLO (bboxes 2D normalisées) pour le dataset."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import carla  # noqa: F401

# Mapping des classes documentées dans le README racine.
YOLO_CLASS_MAPPING: dict[str, int] = {
    "vehicle": 0,
    "walker": 1,
    "traffic_light": 2,
}

# Type d'une ligne de label: (class_id, x_center, y_center, width, height) normalisés [0, 1]
YoloLabel = tuple[int, float, float, float, float]


class YoloLabeler:
    """Calcule les labels YOLO à partir des actors CARLA dans le champ caméra."""

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
        """Retourne les labels YOLO pour la frame courante.

        TODO (Franck): implémenter la projection 3D->2D des actors visibles via
        la matrice intrinsèque CARLA (cf. carla.Sensor.calibration et
        world_to_camera_matrix). Filtrer:
        - Distance ego->actor > max_distance_m
        - Actor hors du champ caméra
        - Actor occlus (optionnel, peut être skip pour le squelette)
        """
        raise NotImplementedError(
            "Projection 3D->2D pas encore implémentée. "
            "À compléter par Franck dans une session ultérieure. "
            "Voir docstring pour la spec."
        )

    def save(self, path: Path) -> None:
        """Compute et écrit les labels au format YOLO."""
        labels = self.compute_labels()
        self.write_labels_to_file(labels, path)

    @staticmethod
    def write_labels_to_file(labels: list[YoloLabel], path: Path) -> None:
        """Écrit une liste de labels (déjà calculés) au format YOLO standard.

        Méthode statique pour être testable sans CARLA.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
            for (cls_id, x, y, w, h) in labels
        ]
        path.write_text("\n".join(lines) + ("\n" if lines else ""))
