"""Contrats de données et protocoles pour les modules de perception.

Tout module dans `src/perception/` doit implémenter les protocoles définis ici.
Cela garantit l'interchangeabilité entre les implémentations réelles
(YOLO entraîné, MIDAS) et les stubs basés sur les ground truths CARLA.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

# ---------------------------------------------------------------------------
# Détection d'objets (YOLO)
# ---------------------------------------------------------------------------


class ObjectClass(str, Enum):
    """Classes d'objets reconnues par le système.

    Aligné sur les 14 classes du dataset CARLA enrichi (enrich_labels.py).
    """

    VEHICLE = "vehicle"
    WALKER = "walker"
    RED_LIGHT = "red_light"
    YELLOW_LIGHT = "yellow_light"
    GREEN_LIGHT = "green_light"
    SPEED_30 = "speed_30"
    SPEED_40 = "speed_40"
    SPEED_60 = "speed_60"
    SPEED_90 = "speed_90"
    STOP = "stop"
    YIELD = "yield"
    UNKNOWN = "unknown"


@dataclass
class DetectedObject:
    """Un objet détecté dans une image."""

    class_name: ObjectClass
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) en pixels
    confidence: float  # [0, 1]
    distance_m: float | None = (
        None  # rempli après fusion avec depth, None si non calculé
    )


class ObjectDetector(Protocol):
    """Contrat pour tout détecteur d'objets (YOLO ou stub CARLA)."""

    def detect(self, image: np.ndarray) -> list[DetectedObject]:
        """Détecte les objets dans une image RGB.

        Args:
            image: image RGB de forme (H, W, 3), dtype uint8.

        Returns:
            Liste d'objets détectés (peut être vide).
        """
        ...


# ---------------------------------------------------------------------------
# Estimation de profondeur (MIDAS / Depth Anything)
# ---------------------------------------------------------------------------


class DepthEstimator(Protocol):
    """Contrat pour tout estimateur de profondeur."""

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Estime la profondeur de chaque pixel d'une image RGB.

        Args:
            image: image RGB de forme (H, W, 3), dtype uint8.

        Returns:
            Depth map de forme (H, W), dtype float32, en mètres.
        """
        ...


# ---------------------------------------------------------------------------
# Détection de lignes
# ---------------------------------------------------------------------------


@dataclass
class Line:
    """Une ligne détectée dans l'image (extrémités en pixels)."""

    start: tuple[int, int]  # (x, y)
    end: tuple[int, int]


@dataclass
class LanesInfo:
    """Informations sur les lignes de la voie courante."""

    left_line: Line | None
    right_line: Line | None
    center_offset: (
        float | None
    )  # offset normalisé [-1, 1] du véhicule par rapport au centre


class LaneDetector(Protocol):
    """Contrat pour tout détecteur de lignes."""

    def detect(self, image: np.ndarray) -> LanesInfo:
        """Détecte les lignes de voie dans une image RGB.

        Args:
            image: image RGB de forme (H, W, 3), dtype uint8.

        Returns:
            Informations sur les lignes (peut contenir des champs None si non détectés).
        """
        ...
