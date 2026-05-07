"""Contrats de données et protocoles pour l'IA centrale.

L'IA centrale consomme :
  - une représentation de la scène (`SceneState`) produite par fusion des
    sorties des modules de perception et de l'état du véhicule
  - une `HighLevelCommand` issue de la navigation
Et produit :
  - un `ControlOutput` (steer, throttle, brake) appliqué au véhicule via CARLA.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.interfaces.navigation_types import HighLevelCommand
from src.interfaces.perception_types import DetectedObject, LanesInfo


@dataclass
class VehicleState:
    """État dynamique du véhicule, lu depuis CARLA à chaque tick."""

    speed_kmh: float
    steer: float  # [-1, 1] courant
    throttle: float  # [0, 1] courant
    brake: float  # [0, 1] courant


@dataclass
class SceneState:
    """Représentation structurée de l'environnement à un instant donné.

    Construit à chaque tick par fusion des sorties des modules de perception
    avec l'état véhicule et la commande haut niveau de la navigation.

    L'IA centrale décide ensuite des contrôles à appliquer à partir de cet
    objet et, en mode end-to-end, de l'image RGB brute.
    """

    image_rgb: np.ndarray  # (H, W, 3) uint8
    objects: list[DetectedObject] = field(default_factory=list)
    depth_map: np.ndarray | None = None  # (H, W) float32, mètres
    lanes: LanesInfo | None = None
    vehicle_state: VehicleState | None = None
    high_level_command: HighLevelCommand = HighLevelCommand.LANE_FOLLOW


@dataclass
class ControlOutput:
    """Décision finale de l'IA centrale, appliquée au véhicule via CARLA."""

    steer: float  # [-1, 1]
    throttle: float  # [0, 1]
    brake: float  # [0, 1]
    reverse: bool = False
