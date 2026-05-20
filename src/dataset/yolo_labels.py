"""YOLO label generation (normalized 2D bboxes) for the dataset.

Outputs a "raw" 3-class scheme : vehicle / walker / traffic_light. La couleur
des feux (rouge / orange / vert) est attribuée hors-ligne par
``src/dataset/colorize_traffic_lights.py``, qui lit ces labels et l'image RGB
pour produire le mapping 5-classes utilisé à l'entraînement YOLO.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.dataset.instance_capture import InstanceCapture, pack_instance_carla

if TYPE_CHECKING:
    import carla  # noqa: F401

# Classes telles qu'écrites par le collector. La couleur des feux n'est pas
# connue ici : voir colorize_traffic_lights.py pour l'expansion en 5 classes.
YOLO_CLASS_MAPPING: dict[str, int] = {
    "vehicle": 0,
    "walker": 1,
    "traffic_light": 2,
}

# CityScape class_id (canal R de l'instance mask CARLA) → classe YOLO du projet.
# Note : Rider (13) est mappé vers vehicle (et non walker) car un motard/cycliste
# se comporte comme un véhicule du point de vue de la conduite (trajectoire,
# vitesse, respect des feux).
_CARLA_TO_YOLO: dict[int, int] = {
    7: 2,   # TrafficLight (couleur résolue offline par colorize_traffic_lights)
    12: 1,  # Pedestrian → walker
    13: 0,  # Rider      → vehicle
    14: 0,  # Car
    15: 0,  # Truck
    16: 0,  # Bus
    18: 0,  # Motorcycle
    19: 0,  # Bicycle
}

_MIN_BBOX_SIDE_PX = 6
# Pour les feux on utilise les composants connexes (semantic mask) au lieu
# de l'instance mask, donc on peut être plus strict sur la taille minimale
# d'un blob (filtre feux fragmentés derrière la végétation).
_MIN_TL_BBOX_SIDE_PX = 4
_MIN_TL_PIXELS = 25
_CARLA_TL_CLASS_ID = 7

# Fill ratio = pixels du mask / aire du bbox. Un vrai véhicule remplit >50%
# de son bbox ; les fuites de l'instance sensor à travers grilles / vitres
# (bug CARLA connu) donnent des clusters épars à <20%. Seuil prudent.
_MIN_FILL_RATIO_NON_TL = 0.25
# Lignes du bas de l'image qu'on considère comme appartenant au capot ego
# (filet de sécurité au cas où la sous-géométrie du capot ait un actor.id
# différent de self.ego.id).
_EGO_HOOD_BOTTOM_MARGIN_PX = 5

# (class_id, x_center, y_center, width, height) — normalized [0, 1].
YoloLabel = tuple[int, float, float, float, float]


class YoloLabeler:
    """Compute YOLO labels from CARLA actors visible in the camera frame."""

    def __init__(
        self,
        world: "carla.World",
        ego: "carla.Vehicle",
        camera_sensor: "carla.Sensor",
        instance_capture: InstanceCapture,
        image_w: int = 1280,
        image_h: int = 720,
        max_distance_m: float = 80.0,
    ) -> None:
        self.world = world
        self.ego = ego
        self.camera_sensor = camera_sensor
        self.instance_capture = instance_capture
        self.image_w = image_w
        self.image_h = image_h
        self.max_distance_m = max_distance_m

    def compute_labels(self) -> list[YoloLabel]:
        """Return YOLO labels for the current frame.

        Reads the latest instance buffer from ``InstanceCapture``, unpacks the
        ``(class_id, instance_id)`` channels, iterates each unique instance to
        derive its 2D bbox via ``np.where``, filters by class / size / distance,
        and emits a normalized YOLO row.
        """
        rgb = self.instance_capture._last_rgb
        if rgb is None:
            return []

        packed = pack_instance_carla(rgb)
        class_map = ((packed >> 16) & 0xFF).astype(np.uint8)
        instance_map = (packed & 0xFFFF).astype(np.uint32)

        labels: list[YoloLabel] = []
        ego_location = self.ego.get_location()
        # L'ego apparaît dans le mask via le capot visible depuis la caméra
        # interne. On l'exclut pour ne pas le labelliser comme un autre véhicule.
        ego_trunc_id = self.ego.id & 0xFFFF

        # Les feux sont traités via composants connexes sur le mask sémantique :
        # l'instance mask est non fiable pour eux (UE4 sub-meshes, instance_id
        # souvent 0 ou pointant vers des actors Python-inaccessibles).
        tl_class_mask = (class_map == _CARLA_TL_CLASS_ID).astype(np.uint8)
        if tl_class_mask.any():
            n_comp, comp_labels = cv2.connectedComponents(tl_class_mask, connectivity=8)
            for comp_id in range(1, n_comp):
                blob = comp_labels == comp_id
                pixel_count = int(blob.sum())
                if pixel_count < _MIN_TL_PIXELS:
                    continue
                ys, xs = np.where(blob)
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
                w_px = x2 - x1
                h_px = y2 - y1
                if w_px < _MIN_TL_BBOX_SIDE_PX or h_px < _MIN_TL_BBOX_SIDE_PX:
                    continue
                tl_yolo_class = _CARLA_TO_YOLO[_CARLA_TL_CLASS_ID]
                labels.append(
                    (
                        tl_yolo_class,
                        float(np.clip((x1 + x2) / 2.0 / self.image_w, 0.0, 1.0)),
                        float(np.clip((y1 + y2) / 2.0 / self.image_h, 0.0, 1.0)),
                        float(np.clip(w_px / self.image_w, 0.0, 1.0)),
                        float(np.clip(h_px / self.image_h, 0.0, 1.0)),
                    )
                )

        for inst_id in np.unique(instance_map):
            if inst_id == 0 or int(inst_id) == ego_trunc_id:
                continue

            mask = instance_map == inst_id
            class_id = int(class_map[mask][0])
            if class_id == _CARLA_TL_CLASS_ID:
                continue  # déjà traité via composants connexes
            yolo_class = _CARLA_TO_YOLO.get(class_id)
            if yolo_class is None:
                continue

            ys, xs = np.where(mask)
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            w_px = x2 - x1
            h_px = y2 - y1
            if w_px < _MIN_BBOX_SIDE_PX or h_px < _MIN_BBOX_SIDE_PX:
                continue

            # Filtre fuite instance-sensor (grilles / vitres semi-transparentes).
            fill_ratio = float(mask.sum()) / max(1, w_px * h_px)
            if fill_ratio < _MIN_FILL_RATIO_NON_TL:
                continue
            # Capot ego : la bbox descend au bord bas de l'image.
            if y2 >= self.image_h - _EGO_HOOD_BOTTOM_MARGIN_PX:
                continue

            actor = self.world.get_actor(int(inst_id))
            if actor is not None:
                distance = actor.get_location().distance(ego_location)
                if distance > self.max_distance_m:
                    continue

            x_center = float(np.clip((x1 + x2) / 2.0 / self.image_w, 0.0, 1.0))
            y_center = float(np.clip((y1 + y2) / 2.0 / self.image_h, 0.0, 1.0))
            w_norm = float(np.clip(w_px / self.image_w, 0.0, 1.0))
            h_norm = float(np.clip(h_px / self.image_h, 0.0, 1.0))

            labels.append((yolo_class, x_center, y_center, w_norm, h_norm))

        return labels

    def save(self, path: Path) -> None:
        labels = self.compute_labels()
        self.write_labels_to_file(labels, path)

    @staticmethod
    def write_labels_to_file(labels: list[YoloLabel], path: Path) -> None:
        """Write labels in YOLO format. Static method so the format is testable without CARLA."""
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
            for (cls_id, x, y, w, h) in labels
        ]
        path.write_text("\n".join(lines) + ("\n" if lines else ""))
