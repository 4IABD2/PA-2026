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
    def __init__(self, world: "carla.World", ego: "carla.Vehicle") -> None:
        self.world = world
        self.ego = ego
        self._planner = self._try_create_planner()

    def _try_create_planner(self) -> object | None:
        try:
            from agents.navigation.local_planner import LocalPlanner

            return LocalPlanner(self.ego)
        except Exception:
            return None

    def current_command(self) -> HighLevelCommand:
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
