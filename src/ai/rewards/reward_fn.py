"""Phase 1 RL reward function — pure function, no CARLA dependency."""

from __future__ import annotations

MAX_SPEED_KMH = 90.0

_W_SPEED = 0.5
_W_CENTER = 0.3
_W_ALIVE = 0.01
_P_OFFROAD = -0.5
_P_COLLISION = -1.0
_P_STALL = -0.05       # breaks the lazy-policy attractor (staying still = 0 risk)
_P_OFF_ROUTE = -0.5    # leaving the planned GPS route is penalised as hard as going off-road


def compute_reward(
    speed_kmh: float,
    center_offset: float,
    is_on_road: bool,
    collision: bool,
    off_route: bool = False,
    max_speed_kmh: float = MAX_SPEED_KMH,
) -> tuple[float, bool]:
    """Compute the per-step reward and whether the episode should terminate.

    All inputs come from CarlaEnv.step() and its perception stubs.

    Args:
        speed_kmh:     Current vehicle speed in km/h.
        center_offset: Lateral deviation from lane centre, normalised to [-1, 1].
                       0 = centred, ±1 = at lane edge. From CarlaGTLaneDetector.
        is_on_road:    True if the vehicle is on a drivable surface.
                       Hardcoded True until Karim's segmentation model is plugged in.
        collision:     True if a collision event was fired this step.
        off_route:     True if the vehicle is more than _OFF_ROUTE_M metres from the
                       nearest waypoint of the planned route (CarlaEnv._is_off_route).
        max_speed_kmh: Target speed ceiling for normalisation. Default = MAX_SPEED_KMH.

    Returns:
        (reward, terminated) where terminated is True only on collision.

    Reward components:
        r_speed     = (speed_kmh / max_speed_kmh) × 0.5    →  [0,   0.5 ]
        r_center    = (1 − |center_offset|) × 0.3           →  [0,   0.3 ]
        r_alive     = +0.01 / step                          →  fixed
        r_stall     = −0.05 if speed < 1 km/h              →  breaks lazy policy
        r_offroad   = −0.5  if not is_on_road              →  off-road penalty
        r_off_route = −0.5  if off_route                   →  GPS deviation penalty
        r_collision = −1.0 + done=True                     →  terminal
    """
    if collision:
        return _P_COLLISION, True

    r_speed = (speed_kmh / max_speed_kmh) * _W_SPEED
    r_center = (1.0 - abs(center_offset)) * _W_CENTER
    r_offroad = 0.0 if is_on_road else _P_OFFROAD
    r_stall = _P_STALL if speed_kmh < 1.0 else 0.0
    r_off_route = _P_OFF_ROUTE if off_route else 0.0

    return r_speed + r_center + _W_ALIVE + r_offroad + r_stall + r_off_route, False
