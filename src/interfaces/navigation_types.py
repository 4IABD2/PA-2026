"""Contrats de données et protocoles pour le module de navigation.

Le module `src/navigation/` produit (1) une route globale via A* sur les waypoints
CARLA et (2) à chaque tick, la prochaine commande haut niveau (gauche, droite,
tout droit, suivre la voie) destinée à l'IA centrale.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class HighLevelCommand(str, Enum):
    """Commande de haut niveau destinée à l'IA centrale.

    Représente l'intention de navigation à la prochaine intersection ou
    le comportement par défaut sur une portion de route droite.
    """

    LEFT = "left"
    RIGHT = "right"
    STRAIGHT = "straight"
    LANE_FOLLOW = "lane_follow"


@dataclass
class Waypoint:
    """Un waypoint sur la route, en coordonnées CARLA monde."""

    x: float
    y: float
    z: float
    yaw_deg: float


@dataclass
class Route:
    """Route globale du véhicule, du point courant à la destination."""

    waypoints: list[Waypoint]
    destination: Waypoint


class RoutePlanner(Protocol):
    """Contrat pour tout planificateur de route (A* sur waypoints CARLA)."""

    def plan(self, start: Waypoint, destination: Waypoint) -> Route:
        """Calcule une route entre deux waypoints.

        Args:
            start: waypoint de départ.
            destination: waypoint d'arrivée.

        Returns:
            Route avec la liste ordonnée des waypoints à suivre.
        """
        ...


class CommandPlanner(Protocol):
    """Contrat pour tout module qui produit la commande haut niveau courante."""

    def next_command(
        self, vehicle_position: Waypoint, route: Route
    ) -> HighLevelCommand:
        """Détermine la commande haut niveau pour le prochain segment.

        Args:
            vehicle_position: position courante du véhicule.
            route: route globale planifiée.

        Returns:
            Commande à transmettre à l'IA centrale.
        """
        ...
