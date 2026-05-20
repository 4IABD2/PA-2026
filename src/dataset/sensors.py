"""Wrapper unifié pour les capteurs caméra CARLA.

Une seule classe ``CameraSensor`` paramétrée par :
- le blueprint CARLA (rgb / depth / semantic_segmentation / instance_segmentation)
- une fonction ``writer(rgb, run_dir, frame_id)`` qui décide où et comment
  écrire les fichiers (1 ou 2 fichiers : .npy + viz PNG par exemple).

Les 4 writers correspondants sont définis ici. Toutes les transformations
pures (decode/pack/colorize) sont dans ``encodings.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, TYPE_CHECKING

import carla
import numpy as np
from PIL import Image

from src.dataset.encodings import (
    CAMERA_LOCATION,
    CAMERA_ROTATION_PITCH,
    colorize_instance,
    colorize_semantic,
    decode_carla_depth,
    decode_semantic_carla,
    pack_instance_carla,
)

if TYPE_CHECKING:
    pass

Writer = Callable[[np.ndarray, Path, int], None]


class CameraSensor:
    """Wrap un capteur caméra-like CARLA : attach / listen / buffer / save.

    Le ``writer`` est invoqué dans ``save`` ; il reçoit l'``rgb`` du dernier
    frame (toujours format RGB uint8, même pour les capteurs depth/semantic/
    instance dont les 3 canaux encodent autre chose qu'une couleur).
    """

    def __init__(
        self,
        world: "carla.World",
        ego: "carla.Vehicle",
        blueprint: str,
        writer: Writer,
        width: int = 1280,
        height: int = 720,
        fov: int = 90,
    ) -> None:
        self.world = world
        self.ego = ego
        self.blueprint = blueprint
        self.writer = writer
        self.width = width
        self.height = height
        self.fov = fov
        self._sensor: "carla.Sensor | None" = None
        self._last_rgb: np.ndarray | None = None
        # Numéro de frame CARLA de la dernière image reçue. Utilisé par le
        # collector pour synchroniser tous les sensors sur le même tick avant
        # la sauvegarde (les callbacks tournent sur des threads séparés).
        self._last_frame_num: int = -1

    def attach(self) -> "carla.Sensor":
        bp = self.world.get_blueprint_library().find(self.blueprint)
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
        self._last_frame_num = image.frame

    def has_frame(self, expected_frame: int) -> bool:
        """True si le buffer correspond à la frame CARLA attendue."""
        return self._last_frame_num == expected_frame

    def save(self, run_dir: Path, frame_id: int) -> None:
        if self._last_rgb is None:
            raise RuntimeError(
                "No buffered image. Call after at least one world.tick()."
            )
        self.writer(self._last_rgb, run_dir, frame_id)

    @property
    def last_rgb(self) -> np.ndarray | None:
        return self._last_rgb


# ----------------------------------------------------------------------------
# Writers — un par modalité. Chacun décide où écrire (1 ou 2 fichiers).
# ----------------------------------------------------------------------------


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_rgb(rgb: np.ndarray, run_dir: Path, frame_id: int) -> None:
    """images/<frame>.jpg (qualité 90)."""
    out = run_dir / "images" / f"{frame_id:06d}.jpg"
    _ensure_dir(out)
    Image.fromarray(rgb).save(str(out), quality=90)


def write_depth(rgb: np.ndarray, run_dir: Path, frame_id: int) -> None:
    """depth/<frame>.npy (float32, mètres, plafonné à 100m)."""
    out = run_dir / "depth" / f"{frame_id:06d}.npy"
    _ensure_dir(out)
    np.save(out, decode_carla_depth(rgb, max_depth_m=100.0))


def write_semantic(rgb: np.ndarray, run_dir: Path, frame_id: int) -> None:
    """semantic/<frame>.npy (uint8 class_id) + viz/semantic/<frame>.png."""
    class_ids = decode_semantic_carla(rgb)
    npy_path = run_dir / "semantic" / f"{frame_id:06d}.npy"
    viz_path = run_dir / "viz" / "semantic" / f"{frame_id:06d}.png"
    _ensure_dir(npy_path)
    _ensure_dir(viz_path)
    np.save(npy_path, class_ids)
    Image.fromarray(colorize_semantic(class_ids)).save(str(viz_path))


def write_instance(rgb: np.ndarray, run_dir: Path, frame_id: int) -> None:
    """instance/<frame>.npy (uint32 packed) + viz/instance/<frame>.png."""
    packed = pack_instance_carla(rgb)
    npy_path = run_dir / "instance" / f"{frame_id:06d}.npy"
    viz_path = run_dir / "viz" / "instance" / f"{frame_id:06d}.png"
    _ensure_dir(npy_path)
    _ensure_dir(viz_path)
    np.save(npy_path, packed)
    Image.fromarray(colorize_instance(packed)).save(str(viz_path))


# Spec déclarative pour le collector : (key, blueprint, writer).
SENSOR_SPECS: list[tuple[str, str, Writer]] = [
    ("rgb",      "sensor.camera.rgb",                   write_rgb),
    ("depth",    "sensor.camera.depth",                 write_depth),
    ("semantic", "sensor.camera.semantic_segmentation", write_semantic),
    ("instance", "sensor.camera.instance_segmentation", write_instance),
]
