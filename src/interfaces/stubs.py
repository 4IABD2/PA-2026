"""Implémentations 'triche' des protocoles de perception, basées sur les
ground truths CARLA.

Ces stubs servent à développer un module en isolation, sans attendre que les
implémentations réelles des coéquipiers (YOLO, MIDAS, lignes) soient prêtes.

Ils LISENT directement l'état interne de CARLA (liste des actors, capteur depth
sémantique...) au lieu de faire de la vraie computer vision sur l'image.

⚠️ NE JAMAIS UTILISER EN PRODUCTION : ces stubs sont des outils de
développement. La promesse du projet est de piloter à partir de la caméra RGB
seule. Voir le README racine section "Conventions architecturales".
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
    import carla  # noqa: F401  (import différé pour permettre les tests sans carla)


class CarlaGTObjectDetector:
    """Détecteur d'objets qui lit la liste des actors CARLA au lieu de faire de la vision.

    Permet de bosser sur l'IA centrale ou la fusion sans attendre YOLO.
    """

    def __init__(self, world: "carla.World", ego_vehicle: "carla.Actor") -> None:
        self.world = world
        self.ego = ego_vehicle

    def detect(self, image: np.ndarray) -> list[DetectedObject]:
        """Ignore l'image, lit `world.get_actors()` et projette en 2D.

        TODO: implémenter la projection 3D->2D des bounding boxes des actors
        sur la caméra ego (utiliser la matrice de projection CARLA).
        """
        raise NotImplementedError("Stub à compléter — voir TODO ci-dessus.")


class CarlaGTDepthEstimator:
    """Estimateur de profondeur qui lit la depth map ground truth de CARLA.

    Utilise un capteur `sensor.camera.depth` attaché au véhicule.
    Permet de bosser sans attendre MIDAS / Depth Anything.
    """

    def __init__(self, depth_sensor: "carla.Sensor") -> None:
        self.depth_sensor = depth_sensor
        self._last_depth: np.ndarray | None = None

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Ignore l'image, retourne la dernière depth map reçue du capteur GT."""
        if self._last_depth is None:
            raise RuntimeError(
                "Aucune frame depth GT reçue. Le capteur depth est-il bien démarré ?"
            )
        return self._last_depth


class CarlaGTLaneDetector:
    """Détecteur de lignes qui calcule l'offset à partir de la position véhicule
    et du waypoint le plus proche dans CARLA.

    Permet de bosser sans attendre la détection de lignes OpenCV.
    """

    def __init__(self, world: "carla.World", ego_vehicle: "carla.Actor") -> None:
        self.world = world
        self.ego = ego_vehicle

    def detect(self, image: np.ndarray) -> LanesInfo:
        """Ignore l'image, calcule l'offset via les waypoints CARLA.

        TODO: implémenter le calcul d'offset entre la position véhicule et le
        waypoint courant, et fabriquer des Line approximatives à partir des
        waypoints adjacents.
        """
        raise NotImplementedError("Stub à compléter — voir TODO ci-dessus.")


__all__ = [
    "CarlaGTObjectDetector",
    "CarlaGTDepthEstimator",
    "CarlaGTLaneDetector",
]
