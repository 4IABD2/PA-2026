"""Wrapper autopilot CARLA et extraction des contrôles experts pour le manifest."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import carla  # noqa: F401


@dataclass(frozen=True)
class ExpertControls:
    """Snapshot des contrôles appliqués par l'expert (autopilot CARLA) sur une frame."""

    steer: float  # [-1, 1]
    throttle: float  # [0, 1]
    brake: float  # [0, 1]
    speed_kmh: float


class ExpertDriver:
    """Wrapper de l'autopilot CARLA + extraction des contrôles appliqués."""

    def __init__(
        self,
        ego: "carla.Vehicle",
        traffic_manager: "carla.TrafficManager",
    ) -> None:
        self.ego = ego
        self.traffic_manager = traffic_manager
        # Active l'autopilot via le Traffic Manager (plus contrôlable que set_autopilot(True))
        ego.set_autopilot(True, traffic_manager.get_port())

    def read_controls(self) -> ExpertControls:
        """Lit les contrôles courants + vitesse du véhicule ego."""
        control = self.ego.get_control()
        velocity = self.ego.get_velocity()
        # |v| en m/s -> km/h
        speed_mps = (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5
        speed_kmh = speed_mps * 3.6

        return ExpertControls(
            steer=float(control.steer),
            throttle=float(control.throttle),
            brake=float(control.brake),
            speed_kmh=float(speed_kmh),
        )
