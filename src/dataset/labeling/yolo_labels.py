from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.dataset.collection.sensors import CameraSensor
from src.dataset.encodings import pack_instance_carla

if TYPE_CHECKING:
    import carla  # noqa: F401

YOLO_CLASS_MAPPING: dict[str, int] = {
    "vehicle": 0,
    "walker": 1,
    "traffic_light": 2,
}

_CARLA_TO_YOLO: dict[int, int] = {
    7: 2,
    12: 1,
    13: 0,
    14: 0,
    15: 0,
    16: 0,
    18: 0,
    19: 0,
}

_SPEED_VALUES = (30, 40, 60, 90)
_SPEED_TO_FINAL = {v: 5 + i for i, v in enumerate(_SPEED_VALUES)}
_SIGN_TYPE_TO_FINAL = {"traffic.stop": 9, "traffic.yield": 10}


def _sign_final_class(type_id: str) -> int | None:
    """Classe finale (5..10) d'un acteur panneau, ou None si non pris en charge."""
    if type_id.startswith("traffic.speed_limit."):
        try:
            return _SPEED_TO_FINAL.get(int(type_id.rsplit(".", 1)[-1]))
        except ValueError:
            return None
    return _SIGN_TYPE_TO_FINAL.get(type_id)


_MIN_BBOX_SIDE_PX = 6
_MIN_TL_BBOX_SIDE_PX = 4
_MIN_TL_PIXELS = 25
_CARLA_TL_CLASS_ID = 7

_MIN_SIGN_BBOX_SIDE_PX = 8
_MIN_SIGN_PIXELS = 40
_CARLA_TS_CLASS_ID = 8

_MIN_FILL_RATIO_NON_TL = 0.25
_EGO_HOOD_BOTTOM_MARGIN_PX = 5

YoloLabel = tuple[int, float, float, float, float]


def labels_from_class_blobs(
    class_map: np.ndarray,
    carla_class_id: int,
    *,
    min_pixels: int,
    min_side_px: int,
    image_w: int,
    image_h: int,
) -> list[YoloLabel]:
    out: list[YoloLabel] = []
    class_mask = (class_map == carla_class_id).astype(np.uint8)
    if not class_mask.any():
        return out

    yolo_class = _CARLA_TO_YOLO[carla_class_id]
    n_comp, comp_labels = cv2.connectedComponents(class_mask, connectivity=8)
    for comp_id in range(1, n_comp):
        blob = comp_labels == comp_id
        if int(blob.sum()) < min_pixels:
            continue
        ys, xs = np.where(blob)
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        w_px = x2 - x1
        h_px = y2 - y1
        if w_px < min_side_px or h_px < min_side_px:
            continue
        out.append(
            (
                yolo_class,
                float(np.clip((x1 + x2) / 2.0 / image_w, 0.0, 1.0)),
                float(np.clip((y1 + y2) / 2.0 / image_h, 0.0, 1.0)),
                float(np.clip(w_px / image_w, 0.0, 1.0)),
                float(np.clip(h_px / image_h, 0.0, 1.0)),
            )
        )
    return out


class YoloLabeler:
    """Compute YOLO labels from CARLA actors visible in the camera frame."""

    def __init__(
        self,
        world: "carla.World",
        ego: "carla.Vehicle",
        camera_sensor: "carla.Sensor",
        instance_capture: CameraSensor,
        image_w: int = 1280,
        image_h: int = 720,
        fov: int = 90,
        max_distance_m: float = 80.0,
    ) -> None:
        self.world = world
        self.ego = ego
        self.camera_sensor = camera_sensor
        self.instance_capture = instance_capture
        self.image_w = image_w
        self.image_h = image_h
        self.fov = fov
        self.max_distance_m = max_distance_m
        self._intrinsics_cache: np.ndarray | None = None

    def _labels_from_connected_components(
        self,
        class_map: np.ndarray,
        carla_class_id: int,
        *,
        min_pixels: int,
        min_side_px: int,
    ) -> list[YoloLabel]:
        return labels_from_class_blobs(
            class_map,
            carla_class_id,
            min_pixels=min_pixels,
            min_side_px=min_side_px,
            image_w=self.image_w,
            image_h=self.image_h,
        )

    def _camera_intrinsics(self) -> np.ndarray:
        """Pinhole intrinsics matrix K from image size + horizontal FOV."""
        if self._intrinsics_cache is None:
            focal = self.image_w / (2.0 * np.tan(np.radians(self.fov) / 2.0))
            self._intrinsics_cache = np.array(
                [
                    [focal, 0.0, self.image_w / 2.0],
                    [0.0, focal, self.image_h / 2.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        return self._intrinsics_cache

    def _project_point(
        self, loc: "carla.Location", world_2_cam: np.ndarray, k: np.ndarray
    ) -> tuple[float, float, float]:
        point = np.array([loc.x, loc.y, loc.z, 1.0])
        cam = world_2_cam @ point
        cam_std = np.array([cam[1], -cam[2], cam[0]])
        depth = float(cam_std[2])
        if depth <= 0:
            return 0.0, 0.0, depth
        img = k @ cam_std
        return float(img[0] / img[2]), float(img[1] / img[2]), depth

    def _sign_blobs(self, class_map: np.ndarray) -> list[tuple[int, int, int, int]]:
        mask = (class_map == _CARLA_TS_CLASS_ID).astype(np.uint8)
        if not mask.any():
            return []
        n, _lbl, stats, _cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
        blobs: list[tuple[int, int, int, int]] = []
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if (
                area < _MIN_SIGN_PIXELS
                or w < _MIN_SIGN_BBOX_SIDE_PX
                or h < _MIN_SIGN_BBOX_SIDE_PX
            ):
                continue
            blobs.append((x, y, x + w, y + h))
        return blobs

    def _projected_signs(
        self, world_2_cam: np.ndarray, k: np.ndarray
    ) -> list[tuple[float, float, int]]:
        import carla

        signs = self.world.get_actors().filter("traffic.*")
        ego_location = self.ego.get_location()
        projected: list[tuple[float, float, int]] = []
        for sign in signs:
            final_cls = _sign_final_class(sign.type_id)
            if final_cls is None:
                continue
            if sign.get_location().distance(ego_location) > self.max_distance_m:
                continue

            verts = sign.bounding_box.get_world_vertices(sign.get_transform())
            cx = sum(v.x for v in verts) / len(verts)
            cy = sum(v.y for v in verts) / len(verts)
            cz = sum(v.z for v in verts) / len(verts)
            u, v_, depth = self._project_point(
                carla.Location(cx, cy, cz), world_2_cam, k
            )
            if depth <= 0:
                continue
            if not (0 <= u < self.image_w and 0 <= v_ < self.image_h):
                continue
            projected.append((u, v_, final_cls))
        return projected

    def _sign_labels(self, class_map: np.ndarray) -> list[YoloLabel]:
        blobs = self._sign_blobs(class_map)
        if not blobs:
            raw_px = int((class_map == _CARLA_TS_CLASS_ID).sum())
            self._sign_debug(0, 0, f"no usable blobs (raw class-8 px={raw_px})")
            return []

        k = self._camera_intrinsics()
        world_2_cam = np.array(self.camera_sensor.get_transform().get_inverse_matrix())
        projected = self._projected_signs(world_2_cam, k)
        if not projected:
            self._sign_debug(len(blobs), 0, "no sign actor projected in frame")
            return []

        pairs: list[tuple[float, int, int]] = []
        for bi, (x1, y1, x2, y2) in enumerate(blobs):
            bcx, bcy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            radius = 4.0 * max(x2 - x1, y2 - y1) + 60.0
            for pi, (u, v_, _cls) in enumerate(projected):
                d = ((bcx - u) ** 2 + (bcy - v_) ** 2) ** 0.5
                if d <= radius:
                    pairs.append((d, bi, pi))
        pairs.sort()

        used_blobs: set[int] = set()
        used_proj: set[int] = set()
        out: list[YoloLabel] = []
        for _d, bi, pi in pairs:
            if bi in used_blobs or pi in used_proj:
                continue
            used_blobs.add(bi)
            used_proj.add(pi)
            x1, y1, x2, y2 = blobs[bi]
            final_cls = projected[pi][2]
            w_px, h_px = x2 - x1, y2 - y1
            out.append(
                (
                    final_cls,
                    float(np.clip((x1 + x2) / 2.0 / self.image_w, 0.0, 1.0)),
                    float(np.clip((y1 + y2) / 2.0 / self.image_h, 0.0, 1.0)),
                    float(np.clip(w_px / self.image_w, 0.0, 1.0)),
                    float(np.clip(h_px / self.image_h, 0.0, 1.0)),
                )
            )
        self._sign_debug(len(blobs), len(projected), f"matched {len(out)}")
        if (
            os.environ.get("CARLA_SIGN_DEBUG") == "1"
            and blobs
            and projected
            and len(out) < min(len(blobs), len(projected))
        ):
            for bi, (x1, y1, x2, y2) in enumerate(blobs):
                if bi in used_blobs:
                    continue
                bcx, bcy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                w, h = x2 - x1, y2 - y1
                best = min(
                    projected, key=lambda p: (p[0] - bcx) ** 2 + (p[1] - bcy) ** 2
                )
                dist = ((best[0] - bcx) ** 2 + (best[1] - bcy) ** 2) ** 0.5
                radius = 4.0 * max(w, h) + 60.0
                print(
                    f"[sign]   blob#{bi} c=({bcx:.0f},{bcy:.0f}) wh=({w},{h}) "
                    f"-> nearest proj=({best[0]:.0f},{best[1]:.0f}) "
                    f"dist={dist:.0f} (radius={radius:.0f})",
                    flush=True,
                )
        return out

    @staticmethod
    def _sign_debug(n_blobs: int, n_proj: int, msg: str) -> None:
        if os.environ.get("CARLA_SIGN_DEBUG") == "1":
            print(f"[sign] blobs={n_blobs} proj={n_proj} :: {msg}", flush=True)

    def compute_labels(self) -> list[YoloLabel]:
        rgb = self.instance_capture.last_rgb
        if rgb is None:
            return []

        packed = pack_instance_carla(rgb)
        class_map = ((packed >> 16) & 0xFF).astype(np.uint8)
        instance_map = (packed & 0xFFFF).astype(np.uint32)

        labels: list[YoloLabel] = []
        ego_location = self.ego.get_location()
        ego_trunc_id = self.ego.id & 0xFFFF

        labels.extend(
            self._labels_from_connected_components(
                class_map,
                _CARLA_TL_CLASS_ID,
                min_pixels=_MIN_TL_PIXELS,
                min_side_px=_MIN_TL_BBOX_SIDE_PX,
            )
        )
        labels.extend(self._sign_labels(class_map))

        for inst_id in np.unique(instance_map):
            if inst_id == 0 or int(inst_id) == ego_trunc_id:
                continue

            mask = instance_map == inst_id
            class_id = int(class_map[mask][0])
            if class_id in (_CARLA_TL_CLASS_ID, _CARLA_TS_CLASS_ID):
                continue
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

            fill_ratio = float(mask.sum()) / max(1, w_px * h_px)
            if fill_ratio < _MIN_FILL_RATIO_NON_TL:
                continue
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
