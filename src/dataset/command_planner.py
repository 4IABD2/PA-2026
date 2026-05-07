"""Calcul de la commande haut niveau pour le manifest dataset.

Mappe la prochaine RoadOption du LocalPlanner CARLA vers une enum compacte
{LEFT, RIGHT, STRAIGHT, LANE_FOLLOW} consommée par l'IA centrale.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import carla  # noqa: F401


class HighLevelCommand(str, Enum):
    """Commande haut niveau exposée dans le manifest.csv."""

    LEFT = "left"
    RIGHT = "right"
    STRAIGHT = "straight"
    LANE_FOLLOW = "lane_follow"


class CommandPlanner:
    """Mappe la prochaine RoadOption du LocalPlanner CARLA vers HighLevelCommand.

    Fallback: si le LocalPlanner n'est pas branchable (problème d'import,
    map sans waypoints, etc.), retourne LANE_FOLLOW par défaut.
    """

    def __init__(self, world: "carla.World", ego: "carla.Vehicle") -> None:
        self.world = world
        self.ego = ego
        self._planner = self._try_create_planner()

    def _try_create_planner(self) -> object | None:
        """Best-effort import et init du LocalPlanner.

        Retourne None si impossible (le current_command() retournera LANE_FOLLOW
        en fallback). À brancher proprement par Franck/Frédéric en session
        suivante quand le besoin métier sera précis.
        """
        try:
            from agents.navigation.local_planner import LocalPlanner

            return LocalPlanner(self.ego)
        except Exception:
            return None

    def current_command(self) -> HighLevelCommand:
        """Retourne la commande haut niveau pour la prochaine action.

        Si le LocalPlanner est dispo et expose une RoadOption, on la mappe.
        Sinon, fallback LANE_FOLLOW (cas par défaut le plus fréquent).
        """
        if self._planner is None:
            return HighLevelCommand.LANE_FOLLOW

        try:
            road_option = getattr(self._planner, "_target_road_option", None)
            if road_option is None:
                return HighLevelCommand.LANE_FOLLOW

            name = str(getattr(road_option, "name", "")).upper()
            if name in {"LEFT", "CHANGELANELEFT"}:
                return HighLevelCommand.LEFT
            if name in {"RIGHT", "CHANGELANERIGHT"}:
                return HighLevelCommand.RIGHT
            if name == "STRAIGHT":
                return HighLevelCommand.STRAIGHT
            return HighLevelCommand.LANE_FOLLOW
        except Exception:
            return HighLevelCommand.LANE_FOLLOW
