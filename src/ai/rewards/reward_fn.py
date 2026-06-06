"""Phase 1 RL reward function — pure function, no CARLA dependency."""

from __future__ import annotations

MAX_SPEED_KMH = 50.0

_W_SPEED = 0.5
_W_CENTER = 0.3
_W_ALIVE = 0.01
_P_OFFROAD = -0.5
_P_COLLISION = -1.0


def compute_reward(
    speed_kmh: float,
    center_offset: float,
    is_on_road: bool,
    collision: bool,
    max_speed_kmh: float = MAX_SPEED_KMH,
) -> tuple[float, bool]:
    if collision:
        return _P_COLLISION, True

    r_speed = (speed_kmh / max_speed_kmh) * _W_SPEED
    r_center = (1.0 - abs(center_offset)) * _W_CENTER
    r_offroad = 0.0 if is_on_road else _P_OFFROAD

    return r_speed + r_center + _W_ALIVE + r_offroad, False
