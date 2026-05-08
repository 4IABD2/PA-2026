"""CARLA autopilot wrapper and expert control extraction for the manifest."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import carla  # noqa: F401


@dataclass(frozen=True)
class ExpertControls:
    """Snapshot of the controls applied by the expert (CARLA autopilot) on a frame."""

    steer: float  # [-1, 1]
    throttle: float  # [0, 1]
    brake: float  # [0, 1]
    speed_kmh: float


class ExpertDriver:
    """Wrapper of the CARLA autopilot + extraction of applied controls."""

    def __init__(
        self,
        ego: "carla.Vehicle",
        traffic_manager: "carla.TrafficManager",
    ) -> None:
        self.ego = ego
        self.traffic_manager = traffic_manager
        # Enable autopilot via the Traffic Manager (more controllable than set_autopilot(True))
        ego.set_autopilot(True, traffic_manager.get_port())

    def read_controls(self) -> ExpertControls:
        """Read the current controls + speed of the ego vehicle."""
        control = self.ego.get_control()
        velocity = self.ego.get_velocity()
        # |v| in m/s -> km/h
        speed_mps = (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5
        speed_kmh = speed_mps * 3.6

        return ExpertControls(
            steer=float(control.steer),
            throttle=float(control.throttle),
            brake=float(control.brake),
            speed_kmh=float(speed_kmh),
        )
