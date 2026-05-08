"""High-level command computation for the dataset manifest.

Maps the next CARLA LocalPlanner RoadOption to a compact enum
{LEFT, RIGHT, STRAIGHT, LANE_FOLLOW} consumed by the central AI.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import carla  # noqa: F401


class HighLevelCommand(str, Enum):
    """High-level command exposed in manifest.csv."""

    LEFT = "left"
    RIGHT = "right"
    STRAIGHT = "straight"
    LANE_FOLLOW = "lane_follow"


class CommandPlanner:
    """Map the next CARLA LocalPlanner RoadOption to a HighLevelCommand.

    Fallback: if the LocalPlanner cannot be wired up (import error, map
    without waypoints, etc.), returns LANE_FOLLOW by default.
    """

    def __init__(self, world: "carla.World", ego: "carla.Vehicle") -> None:
        self.world = world
        self.ego = ego
        self._planner = self._try_create_planner()

    def _try_create_planner(self) -> object | None:
        """Best-effort import and init of the LocalPlanner.

        Returns None if not possible (current_command() will fall back to
        LANE_FOLLOW). To be wired up properly by Franck/Frédéric in a later
        session when the business need is precise.
        """
        try:
            from agents.navigation.local_planner import LocalPlanner

            return LocalPlanner(self.ego)
        except Exception:
            return None

    def current_command(self) -> HighLevelCommand:
        """Return the high-level command for the next action.

        If the LocalPlanner is available and exposes a RoadOption, map it.
        Otherwise fall back to LANE_FOLLOW (the most frequent default case).
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
