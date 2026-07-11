"""Phase 1 RL reward function — pure function, no CARLA dependency."""

from __future__ import annotations

MAX_SPEED_KMH = 90.0

_W_PROGRESS = 1.0  # raised from 0.3 -- measured on v10, its real contribution
# was ~0.003/step, negligible next to r_offroad (-0.5/step); not a derived
# optimum, revisit if v11 still shows no measurable behavior change
_W_CENTER = 0.3
_W_ALIVE = 0.05  # raised from 0.01 -- makes mere survival per step meaningfully
# more valuable, a direct (partial) counterweight to ending an episode early
# (runs/AUDIT.md section 2.1, option (a))
_P_OFFROAD = -0.5
_P_COLLISION_BASE = -5.0
_P_COLLISION_SPEED_SCALE = -0.20  # per km/h of speed at the moment of impact
_P_STALL = -0.20  # breaks the lazy-policy attractor (staying still = 0 risk),
# except when stopping is legitimate (red light ahead, vehicle/walker danger) --
# see is_legitimate_stop in compute_reward()
_P_OFF_ROUTE = -0.5  # re-aligned with _P_OFFROAD (both -0.5)

_W_FOLLOWING = 0.2
_SAFE_HEADWAY_S = 2.0  # standard "2-second rule" following distance

_W_WALKER_PROXIMITY = 0.3  # higher than _W_FOLLOWING: pedestrian safety takes priority
_WALKER_DANGER_M = 10.0

_STALL_RED_LIGHT_GATE_M = 15.0  # bigger than _RED_LIGHT_VIOLATION_DIST_M (5.0) --
# stopping well before the violation line must not be punished as stalling

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
    """Penalise following another vehicle with less than the 2-second safe headway.

    Distance-only thresholds don't account for speed (15 m is safe at 10 km/h,
    dangerous at 80 km/h) — this uses estimated time-to-impact instead.
    """
    if speed_kmh < 1.0 or distance_m >= 50.0:
        return 0.0
    headway_s = distance_m / (speed_kmh / 3.6)
    if headway_s >= _SAFE_HEADWAY_S:
        return 0.0
    return -(1.0 - headway_s / _SAFE_HEADWAY_S) * _W_FOLLOWING


def _walker_penalty(distance_m: float) -> float:
    """Penalise getting close to a pedestrian, proportional to proximity.

    Distance-based rather than time-headway: a pedestrian can move
    unpredictably regardless of the ego vehicle's speed.
    """
    if distance_m >= _WALKER_DANGER_M:
        return 0.0
    return -(1.0 - distance_m / _WALKER_DANGER_M) * _W_WALKER_PROXIMITY


def _speeding_penalty(
    speed_kmh: float, speed_limit_kmh: float | None, max_speed_kmh: float
) -> float:
    """Penalise exceeding the detected speed limit, proportional to the overshoot."""
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
    """Small positive signal for driving with no active danger, complementing the
    purely punitive following/walker/speeding penalties above — reuses their
    exact thresholds so "safe" here means precisely "none of those would fire".
    """
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
) -> tuple[float, bool, dict[str, float]]:
    """Compute the per-step reward and whether the episode should terminate.

    All inputs come from CarlaEnv.step(), already converted to physical units
    (metres, km/h) — this function has no knowledge of normalization constants.

    Args:
        speed_kmh:     Current vehicle speed in km/h.
        center_offset: Lateral deviation from lane centre, normalised to [-1, 1].
                       0 = centred, ±1 = at lane edge. From Karim's lane detection
                       (lane_geometry's `offset`, via lane_perception.estimate()).
        is_on_road:    True if the vehicle is on a drivable surface (Karim).
        collision:     True if a collision event was fired this step.
        off_route:     True if the vehicle is more than _OFF_ROUTE_M metres from the
                       nearest waypoint of the planned route (CarlaEnv._is_off_route).
        max_speed_kmh: Target speed ceiling for normalisation. Default = MAX_SPEED_KMH.
        nearest_vehicle_m: Distance in metres to the nearest detected vehicle ahead
                       (Franck). `inf` if none detected within sensor range.
        nearest_walker_m:  Distance in metres to the nearest detected pedestrian
                       (Franck). `inf` if none detected.
        red_light_distance_m: Distance in metres to the nearest detected red
                       light ahead (Franck). `inf` if none detected. Used to
                       exempt r_stall while legitimately stopped at a light,
                       not just to flag the single-step violation.
        speed_limit_kmh:   Last detected speed limit sign value, or None if none
                       has been seen yet this episode (Franck).
        collision_speed_kmh: Vehicle speed at the instant the collision fired.
        red_light_violation:  True on the single step a red light is judged run
                       (CarlaEnv resolves the "already penalised this light" state).
        stop_yield_violation: Same as above, for stop/yield signs.
        reached_destination: True the step the ego is judged to have arrived
                       (CarlaEnv checks both proximity and minimum distance
                       travelled). Terminates the episode like a collision,
                       but with a positive reward.
        steer_delta:   abs(current steer - previous steer), for the jerk
                       penalty. 0.0 (default) means no penalty.
        progress_delta: γ·Φ(s') − Φ(s) for potential-based reward shaping on
                       distance-to-destination, pre-computed by CarlaEnv
                       (Φ(s) = −normalized_distance_to_destination(s)).
                       0.0 (default) means no shaping applied this step.

    Returns:
        (reward, terminated, components) where terminated is True on
        collision or when the destination is reached, and components holds
        every key in REWARD_COMPONENT_KEYS (0.0 for any that didn't
        contribute this step). reward always equals sum(components.values())
        — the two are never computed independently.
    """
    if collision:
        components = {key: 0.0 for key in REWARD_COMPONENT_KEYS}
        components["r_collision"] = (
            _P_COLLISION_BASE + _P_COLLISION_SPEED_SCALE * collision_speed_kmh
        )
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
        "r_center": (1.0 - abs(center_offset)) * _W_CENTER if is_on_road else 0.0,
        "r_alive": _W_ALIVE,
        "r_offroad": 0.0 if is_on_road else _P_OFFROAD,
        "r_stall": (
            0.0 if is_legitimate_stop else (_P_STALL if speed_kmh < 1.0 else 0.0)
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
