from __future__ import annotations

MAX_SPEED_KMH = 90.0

_W_PROGRESS = 1.0
_W_CENTER = 0.3
_W_ALIVE = 0.05
_P_OFFROAD = -0.5
_P_COLLISION_BASE = -5.0
_P_COLLISION_SPEED_SCALE = -0.20
_P_STALL = -0.20
_STALL_RAMP_STEPS = 200
_STALL_MAX_FACTOR = 2.0
_P_OFF_ROUTE = -0.5

_W_FOLLOWING = 0.2
_SAFE_HEADWAY_S = 2.0

_W_WALKER_PROXIMITY = 0.3
_WALKER_DANGER_M = 10.0

_STALL_RED_LIGHT_GATE_M = 15.0

_W_SPEEDING = 0.3
_SPEEDING_TOLERANCE_KMH = 5.0

_P_RED_LIGHT_VIOLATION = -2.0
_P_STOP_YIELD_VIOLATION = -1.0

_P_DEST_REACHED = 10.0
_W_SAFE_DRIVING = 0.05
_W_JERK = 0.1

REWARD_COMPONENT_KEYS: tuple[str, ...] = (
    "r_progress",
    "r_center",
    "r_alive",
    "r_offroad",
    "r_stall",
    "r_off_route",
    "r_following",
    "r_walker",
    "r_speeding",
    "r_red_light",
    "r_stop_yield",
    "r_collision",
    "r_destination",
    "r_safe",
    "r_jerk",
)


def _following_penalty(distance_m: float, speed_kmh: float) -> float:
    if speed_kmh < 1.0 or distance_m >= 50.0:
        return 0.0
    headway_s = distance_m / (speed_kmh / 3.6)
    if headway_s >= _SAFE_HEADWAY_S:
        return 0.0
    return -(1.0 - headway_s / _SAFE_HEADWAY_S) * _W_FOLLOWING


def _walker_penalty(distance_m: float) -> float:
    if distance_m >= _WALKER_DANGER_M:
        return 0.0
    return -(1.0 - distance_m / _WALKER_DANGER_M) * _W_WALKER_PROXIMITY


def _speeding_penalty(
    speed_kmh: float, speed_limit_kmh: float | None, max_speed_kmh: float
) -> float:
    if speed_limit_kmh is None:
        return 0.0
    over = speed_kmh - speed_limit_kmh - _SPEEDING_TOLERANCE_KMH
    if over <= 0:
        return 0.0
    return -(over / max_speed_kmh) * _W_SPEEDING


def _safe_driving_bonus(
    speed_kmh: float,
    nearest_vehicle_m: float,
    nearest_walker_m: float,
    speed_limit_kmh: float | None,
    max_speed_kmh: float,
) -> float:
    following_ok = _following_penalty(nearest_vehicle_m, speed_kmh) == 0.0
    walker_ok = _walker_penalty(nearest_walker_m) == 0.0
    speeding_ok = _speeding_penalty(speed_kmh, speed_limit_kmh, max_speed_kmh) == 0.0
    if following_ok and walker_ok and speeding_ok:
        return _W_SAFE_DRIVING
    return 0.0


def compute_reward(
    speed_kmh: float,
    center_offset: float,
    is_on_road: bool,
    collision: bool,
    off_route: bool = False,
    max_speed_kmh: float = MAX_SPEED_KMH,
    nearest_vehicle_m: float = float("inf"),
    nearest_walker_m: float = float("inf"),
    red_light_distance_m: float = float("inf"),
    speed_limit_kmh: float | None = None,
    collision_speed_kmh: float = 0.0,
    red_light_violation: bool = False,
    stop_yield_violation: bool = False,
    reached_destination: bool = False,
    steer_delta: float = 0.0,
    progress_delta: float = 0.0,
    remaining_frac: float = 0.0,
    stall_steps: int = 0,
) -> tuple[float, bool, dict[str, float]]:
    if collision:
        components = {key: 0.0 for key in REWARD_COMPONENT_KEYS}
        collision_scale = 1.0 + max(0.0, min(1.0, remaining_frac))
        components["r_collision"] = (
            _P_COLLISION_BASE + _P_COLLISION_SPEED_SCALE * collision_speed_kmh
        ) * collision_scale
        return components["r_collision"], True, components

    if reached_destination:
        components = {key: 0.0 for key in REWARD_COMPONENT_KEYS}
        components["r_destination"] = _P_DEST_REACHED
        return components["r_destination"], True, components

    is_legitimate_stop = (
        red_light_distance_m < _STALL_RED_LIGHT_GATE_M
        or nearest_vehicle_m < _WALKER_DANGER_M
        or nearest_walker_m < _WALKER_DANGER_M
    )

    components = {
        "r_progress": progress_delta * _W_PROGRESS,
        "r_center": (
            (1.0 - abs(center_offset)) * _W_CENTER
            if (is_on_road and speed_kmh >= 1.0)
            else 0.0
        ),
        "r_alive": _W_ALIVE,
        "r_offroad": 0.0 if is_on_road else _P_OFFROAD,
        "r_stall": (
            0.0
            if is_legitimate_stop
            else (
                _P_STALL * min(1.0 + stall_steps / _STALL_RAMP_STEPS, _STALL_MAX_FACTOR)
                if speed_kmh < 1.0
                else 0.0
            )
        ),
        "r_off_route": _P_OFF_ROUTE if off_route else 0.0,
        "r_following": _following_penalty(nearest_vehicle_m, speed_kmh),
        "r_walker": _walker_penalty(nearest_walker_m),
        "r_speeding": _speeding_penalty(speed_kmh, speed_limit_kmh, max_speed_kmh),
        "r_red_light": _P_RED_LIGHT_VIOLATION if red_light_violation else 0.0,
        "r_stop_yield": _P_STOP_YIELD_VIOLATION if stop_yield_violation else 0.0,
        "r_collision": 0.0,
        "r_destination": 0.0,
        "r_safe": _safe_driving_bonus(
            speed_kmh,
            nearest_vehicle_m,
            nearest_walker_m,
            speed_limit_kmh,
            max_speed_kmh,
        ),
        "r_jerk": -steer_delta * _W_JERK,
    }
    return sum(components.values()), False, components
