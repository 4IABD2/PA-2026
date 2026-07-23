from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import carla  # noqa: F401


@dataclass(frozen=True)
class ExpertControls:
    """Snapshot of the controls applied by the autopilot on a frame."""

    steer: float
    throttle: float
    brake: float
    speed_kmh: float


class ExpertDriver:
    """Enable CARLA autopilot on the ego (via the Traffic Manager) and read back its controls."""

    def __init__(
        self,
        ego: "carla.Vehicle",
        traffic_manager: "carla.TrafficManager",
    ) -> None:
        self.ego = ego
        self.traffic_manager = traffic_manager
        ego.set_autopilot(True, traffic_manager.get_port())

    def read_controls(self) -> ExpertControls:
        control = self.ego.get_control()
        velocity = self.ego.get_velocity()
        speed_kmh = (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5 * 3.6

        return ExpertControls(
            steer=float(control.steer),
            throttle=float(control.throttle),
            brake=float(control.brake),
            speed_kmh=float(speed_kmh),
        )
