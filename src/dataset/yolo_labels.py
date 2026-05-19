"""YOLO label generation (normalized 2D bboxes) for the dataset.

Outputs a "raw" 4-class scheme :

    0 vehicle, 1 walker, 2 traffic_light, 3 traffic_sign

L'expansion en 12 classes finales (red/yellow/green + speed_30..90) est faite
hors-ligne par ``enrich_labels.py``.

Source des bboxes :
- vehicle / walker : instance mask CARLA (1 instance_id par objet)
- traffic_light / traffic_sign : composants connexes sur le **semantic mask**
  (l'instance mask est non fiable pour ces statiques de map, cf.
  ``PROBLEMS_AND_FIXES.md``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.dataset.encodings import pack_instance_carla
from src.dataset.sensors import CameraSensor

if TYPE_CHECKING:
    import carla  # noqa: F401

YOLO_CLASS_MAPPING: dict[str, int] = {
    "vehicle": 0,
    "walker": 1,
    "traffic_light": 2,
    "traffic_sign": 3,
}

# CityScape class_id (canal R) → classe YOLO du projet.
# Note : Rider (13) → vehicle parce qu'un motard/cycliste se comporte comme un
# véhicule (trajectoire, vitesse, respect des feux), pas comme un piéton.
_CARLA_TO_YOLO: dict[int, int] = {
    12: 1,  # Pedestrian → walker
    13: 0,  # Rider      → vehicle
    14: 0,  # Car
    15: 0,  # Truck
    16: 0,  # Bus
    18: 0,  # Motorcycle
    19: 0,  # Bicycle
}

# Classes statiques traitées via composants connexes (semantic mask).
_CARLA_TL_CLASS_ID = 7   # CityScape : TrafficLight  → 2
_CARLA_SIGN_CLASS_ID = 8  # CityScape : TrafficSign  → 3

# Filtres bbox pour les classes "dynamiques" (vehicle / walker).
_MIN_BBOX_SIDE_PX = 6
_MIN_FILL_RATIO_NON_STATIC = 0.25  # fuites grilles / vitres CARLA → drop
_EGO_HOOD_BOTTOM_MARGIN_PX = 5  # ego sub-mesh capot

# Filtres bbox pour les classes statiques (TL / sign) traitées par CC.
_MIN_TL_BBOX_SIDE_PX = 4
_MIN_TL_PIXELS = 25
_MIN_SIGN_BBOX_SIDE_PX = 5
_MIN_SIGN_PIXELS = 30

# (class_id, x_center, y_center, width, height) — normalized [0, 1].
YoloLabel = tuple[int, float, float, float, float]


class YoloLabeler:
    """Calcule les labels YOLO depuis les masks CARLA du frame courant."""

    def __init__(
        self,
        world: "carla.World",
        ego: "carla.Vehicle",
        instance_sensor: CameraSensor,
        image_w: int = 1280,
        image_h: int = 720,
        max_distance_m: float = 80.0,
    ) -> None:
        self.world = world
        self.ego = ego
        self.instance_sensor = instance_sensor
        self.image_w = image_w
        self.image_h = image_h
        self.max_distance_m = max_distance_m

    def compute_labels(self) -> list[YoloLabel]:
        rgb = self.instance_sensor.last_rgb
        if rgb is None:
            return []

        packed = pack_instance_carla(rgb)
        class_map = ((packed >> 16) & 0xFF).astype(np.uint8)
        instance_map = (packed & 0xFFFF).astype(np.uint32)

        labels: list[YoloLabel] = []

        # --- Statiques (TL, signs) via composants connexes -------------------
        labels.extend(
            self._labels_from_connected_components(
                class_map=class_map,
                target_class_id=_CARLA_TL_CLASS_ID,
                yolo_class=YOLO_CLASS_MAPPING["traffic_light"],
                min_pixels=_MIN_TL_PIXELS,
                min_side=_MIN_TL_BBOX_SIDE_PX,
            )
        )
        labels.extend(
            self._labels_from_connected_components(
                class_map=class_map,
                target_class_id=_CARLA_SIGN_CLASS_ID,
                yolo_class=YOLO_CLASS_MAPPING["traffic_sign"],
                min_pixels=_MIN_SIGN_PIXELS,
                min_side=_MIN_SIGN_BBOX_SIDE_PX,
            )
        )

        # --- Dynamiques (vehicle, walker) via instance_id --------------------
        labels.extend(
            self._labels_from_instances(class_map=class_map, instance_map=instance_map)
        )

        return labels

    # ------------------------------------------------------------------------

    def _labels_from_connected_components(
        self,
        *,
        class_map: np.ndarray,
        target_class_id: int,
        yolo_class: int,
        min_pixels: int,
        min_side: int,
    ) -> list[YoloLabel]:
        """Produit un label par blob connexe de pixels ``class_id == target``."""
        mask = (class_map == target_class_id).astype(np.uint8)
        if not mask.any():
            return []

        n_comp, comp_labels = cv2.connectedComponents(mask, connectivity=8)
        out: list[YoloLabel] = []
        for comp_id in range(1, n_comp):
            blob = comp_labels == comp_id
            if int(blob.sum()) < min_pixels:
                continue
            ys, xs = np.where(blob)
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            w_px = x2 - x1
            h_px = y2 - y1
            if w_px < min_side or h_px < min_side:
                continue
            out.append(self._normalize_bbox(yolo_class, x1, x2, y1, y2))
        return out

    def _labels_from_instances(
        self, *, class_map: np.ndarray, instance_map: np.ndarray
    ) -> list[YoloLabel]:
        """Itère les instance_ids du mask et produit un label par objet
        dynamique (vehicle / walker). Filtres : ego, hors mapping, taille,
        fill ratio (anti-fuite), capot ego (fallback), distance > max."""
        ego_location = self.ego.get_location()
        ego_trunc_id = self.ego.id & 0xFFFF

        out: list[YoloLabel] = []
        for inst_id in np.unique(instance_map):
            if inst_id == 0 or int(inst_id) == ego_trunc_id:
                continue

            mask = instance_map == inst_id
            class_id = int(class_map[mask][0])
            if class_id in (_CARLA_TL_CLASS_ID, _CARLA_SIGN_CLASS_ID):
                continue  # traités par CC
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

            # Fuites instance-sensor à travers grilles / vitres.
            fill_ratio = float(mask.sum()) / max(1, w_px * h_px)
            if fill_ratio < _MIN_FILL_RATIO_NON_STATIC:
                continue
            # Filet de sécurité contre les sous-meshes du capot ego.
            if y2 >= self.image_h - _EGO_HOOD_BOTTOM_MARGIN_PX:
                continue

            actor = self.world.get_actor(int(inst_id))
            if actor is not None:
                if actor.get_location().distance(ego_location) > self.max_distance_m:
                    continue

            out.append(self._normalize_bbox(yolo_class, x1, x2, y1, y2))
        return out

    def _normalize_bbox(
        self, yolo_class: int, x1: int, x2: int, y1: int, y2: int
    ) -> YoloLabel:
        w_px = x2 - x1
        h_px = y2 - y1
        return (
            yolo_class,
            float(np.clip((x1 + x2) / 2.0 / self.image_w, 0.0, 1.0)),
            float(np.clip((y1 + y2) / 2.0 / self.image_h, 0.0, 1.0)),
            float(np.clip(w_px / self.image_w, 0.0, 1.0)),
            float(np.clip(h_px / self.image_h, 0.0, 1.0)),
        )

    # ------------------------------------------------------------------------

    def save(self, path: Path) -> None:
        labels = self.compute_labels()
        self.write_labels_to_file(labels, path)

    @staticmethod
    def write_labels_to_file(labels: list[YoloLabel], path: Path) -> None:
        """Format YOLO : 1 ligne par objet, valeurs normalisées."""
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
            for (cls_id, x, y, w, h) in labels
        ]
        path.write_text("\n".join(lines) + ("\n" if lines else ""))
