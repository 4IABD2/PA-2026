"""Benchmark scenario suite — 13 fixed scenarios for the final demo video.

These scenarios do not change until the final project presentation.
Phase 1 scenarios are evaluated (success_fn).
Phase 2 scenarios are recorded as-is — model is not expected to react yet.

After running scripts/explore_spawns.py --host <ip>, copy the printed
spawn_idx values into the BENCHMARK_SCENARIOS list below.
"""

from __future__ import annotations

import random

import numpy as np

from src.ai.inference.rl_demo import Scenario

# ---------------------------------------------------------------------------
# NPC / walker setup functions
# ---------------------------------------------------------------------------


def _road_waypoint(world, loc):
    """Snap a location to the nearest drivable road waypoint. Returns None if not found."""
    import carla  # noqa: PLC0415

    return world.get_map().get_waypoint(
        loc, project_to_road=True, lane_type=carla.LaneType.Driving
    )


def _spawn_vehicle(world, wp, vehicles=None):
    """Spawn a vehicle at a waypoint transform (+0.5 m Z offset). Returns actor or None."""
    import carla  # noqa: PLC0415

    if vehicles is None:
        vehicles = list(world.get_blueprint_library().filter("vehicle.*"))
    bp = random.choice(vehicles)
    loc = wp.transform.location
    transform = carla.Transform(
        carla.Location(x=loc.x, y=loc.y, z=loc.z + 0.5),
        wp.transform.rotation,
    )
    return world.try_spawn_actor(bp, transform)


def _setup_npc_ahead(world, ego):
    """Slow vehicle on the road ~25 m ahead — tests longitudinal distance keeping."""
    import carla  # noqa: PLC0415

    transform = ego.get_transform()
    fwd = transform.get_forward_vector()

    # Walk along waypoints to find a road point ~25 m ahead
    wp = _road_waypoint(world, transform.location)
    if wp is None:
        return []
    dist = 0.0
    while dist < 25.0:
        nexts = wp.next(3.0)
        if not nexts:
            break
        wp = nexts[0]
        dist += 3.0

    vehicles = list(world.get_blueprint_library().filter("vehicle.*"))
    npc = _spawn_vehicle(world, wp, vehicles)
    if npc:
        npc.apply_control(carla.VehicleControl(throttle=0.18))
        return [npc]
    return []


def _setup_npc_crossing(world, ego):
    """Vehicle on a perpendicular road ~20 m to the right, heading to cross."""
    import carla  # noqa: PLC0415

    transform = ego.get_transform()
    right = transform.get_right_vector()

    # Find a road waypoint to the right of the ego
    for dist_m in (20, 15, 25, 10):
        candidate = carla.Location(
            x=transform.location.x + right.x * dist_m,
            y=transform.location.y + right.y * dist_m,
            z=transform.location.z,
        )
        wp = _road_waypoint(world, candidate)
        if wp is not None:
            break
    else:
        return []

    vehicles = list(world.get_blueprint_library().filter("vehicle.*"))
    npc = _spawn_vehicle(world, wp, vehicles)
    if npc:
        npc.apply_control(carla.VehicleControl(throttle=0.5))
        return [npc]
    return []


def _setup_pedestrian(world, ego):
    """Pedestrian crossing the road 20 m ahead."""
    import carla  # noqa: PLC0415

    transform = ego.get_transform()
    fwd = transform.get_forward_vector()
    right = transform.get_right_vector()
    loc = carla.Location(
        x=transform.location.x + fwd.x * 20 + right.x * 3,
        y=transform.location.y + fwd.y * 20 + right.y * 3,
        z=transform.location.z,
    )
    walkers = list(world.get_blueprint_library().filter("walker.pedestrian.*"))
    if not walkers:
        return []
    bp = random.choice(walkers)
    walker = world.try_spawn_actor(bp, carla.Transform(loc))
    if walker is None:
        return []

    # AI controller to make the walker move across the road
    ctrl_bp = world.get_blueprint_library().find("controller.ai.walker")
    ctrl = world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=walker)
    world.tick()
    ctrl.start()
    dest = carla.Location(
        x=loc.x - right.x * 6,
        y=loc.y - right.y * 6,
        z=loc.z,
    )
    ctrl.go_to_location(dest)
    ctrl.set_max_speed(1.2)
    return [ctrl, walker]


def _setup_emergency_stop(world, ego):
    """Static vehicle on the road ~8 m ahead — obstacle_norm will be very low (~0.16)."""
    import carla  # noqa: PLC0415

    wp = _road_waypoint(world, ego.get_transform().location)
    if wp is None:
        return []
    for _ in range(3):
        nexts = wp.next(3.0)
        if not nexts:
            break
        wp = nexts[0]

    npc = _spawn_vehicle(world, wp)
    if npc:
        npc.apply_control(carla.VehicleControl(hand_brake=True))
        return [npc]
    return []


# ---------------------------------------------------------------------------
# Success functions  (metrics: dict)
#   metrics keys: center_offsets, speeds, rewards, terminated, steps
# ---------------------------------------------------------------------------

_MIN_DIST_M = 25.0  # car must travel at least this far to count as "having driven"


def _success_no_crash(m: dict) -> bool:
    """No collision AND car actually drove (>= 25 m from start)."""
    return not m["terminated"] and m.get("max_dist_from_start", 0.0) >= _MIN_DIST_M


def _success_straight(m: dict) -> bool:
    """Drove >= 25 m, stayed centered, no collision."""
    return (
        not m["terminated"]
        and m.get("max_dist_from_start", 0.0) >= _MIN_DIST_M
        and float(np.mean(np.abs(m["center_offsets"]))) < 0.3
    )


def _success_reached_dest(m: dict) -> bool:
    """Reached the planned GPS destination without crashing."""
    return m.get("reached_dest", False) and not m["terminated"]


# ---------------------------------------------------------------------------
# BENCHMARK_SCENARIOS
#
# spawn_idx values are filled after running:
#     uv run python3 scripts/explore_spawns.py --host <carla-ip>
#
# Town02 — 101 spawn points.
# Last filled: 2026-07-07 via scripts/explore_spawns.py + scripts/find_dest_spawns.py --host localhost
# pedestrian / lane_change : NOT FOUND in Town02 — set manually when available.
# ---------------------------------------------------------------------------

BENCHMARK_SCENARIOS: list[Scenario] = [
    # ── Phase 1 : Lane keeping ────────────────────────────────────────────
    Scenario(
        name="straight",
        description="Straight road - lane centering and speed control",
        spawn_idx=13,
        max_steps=300,
        phase=1,
        expected="stay centered, maintain constant speed",
        success_fn=_success_straight,
    ),
    Scenario(
        name="curve_left",
        description="Left curve - heading correction",
        spawn_idx=35,
        max_steps=300,
        phase=1,
        expected="follow the curve without leaving the lane",
        success_fn=_success_no_crash,
    ),
    Scenario(
        name="curve_right",
        description="Right curve - heading correction",
        spawn_idx=31,
        max_steps=300,
        phase=1,
        expected="follow the curve without leaving the lane",
        success_fn=_success_no_crash,
    ),
    # ── Phase 1 : Navigation ──────────────────────────────────────────────
    #
    # dest_spawn_idx forces a route replan so nav gives the correct command.
    # To find the right dest_spawn_idx:
    #   1. Run `uv run python3 scripts/explore_spawns.py --host <ip>`
    #   2. Look at the map — pick a spawn that is ONLY reachable via the desired maneuver.
    #   3. Verify by running run_eval.py and watching the nav command overlay.
    #
    # success_fn=_success_reached_dest validates that the car actually completed the maneuver
    # (reached the destination), not just that it drove without crashing.
    Scenario(
        name="turn_left",
        description="Junction - turn left on nav command",
        spawn_idx=79,
        max_steps=350,
        phase=1,
        expected="turn left at junction, reach destination",
        dest_spawn_idx=63,  # 30.5 m away, snap dist 14.7 m — verified via find_dest_spawns.py
        target_radius=15.0,
        success_fn=_success_reached_dest,
    ),
    Scenario(
        name="turn_right",
        description="Junction - turn right on nav command",
        spawn_idx=81,
        max_steps=350,
        phase=1,
        expected="turn right at junction, reach destination",
        dest_spawn_idx=49,  # 56.5 m away, snap dist 2.8 m — best right-branch candidate
        # (find_dest_spawns.py's own top pick, idx=85, had snap dist 29.3m —
        # picked this closer-snapped alternative from the same branch instead)
        target_radius=15.0,
        success_fn=_success_reached_dest,
    ),
    Scenario(
        name="junction_straight",
        description="T-junction - go straight on nav command",
        spawn_idx=47,
        max_steps=350,
        phase=1,
        expected="cross the junction straight, reach destination",
        dest_spawn_idx=49,  # 85.6 m away, snap dist 2.8 m — verified via find_dest_spawns.py
        target_radius=15.0,
        success_fn=_success_reached_dest,
    ),
    # ── Phase 1 : Obstacle avoidance ──────────────────────────────────────
    Scenario(
        name="npc_follow",
        description="Follow a slow vehicle ahead",
        spawn_idx=13,
        max_steps=300,
        phase=1,
        expected="slow down, maintain safe following distance",
        setup_fn=_setup_npc_ahead,
        success_fn=_success_no_crash,
    ),
    Scenario(
        name="npc_crossing",
        description="Vehicle crossing the road at junction",
        spawn_idx=79,
        max_steps=300,
        phase=1,
        expected="brake or avoid the crossing vehicle",
        setup_fn=_setup_npc_crossing,
        success_fn=_success_no_crash,
    ),
    # ── Phase 2 : Traffic management ──────────────────────────────────────
    Scenario(
        name="red_light",
        description="Red light - full stop expected",
        spawn_idx=60,
        max_steps=300,
        phase=2,
        expected="full stop before the stop line",
    ),
    Scenario(
        name="speed_zone",
        description="30 km/h speed zone - respect the limit",
        spawn_idx=8,
        max_steps=300,
        phase=2,
        expected="speed <= 30 km/h in the zone",
    ),
    # ── Phase 2 : Pedestrians ────────────────────────────────────────────
    Scenario(
        name="pedestrian",
        description="Pedestrian on crosswalk - full stop expected",
        spawn_idx=0,  # identify manually near a crosswalk
        max_steps=300,
        phase=2,
        expected="full stop in front of pedestrian",
        setup_fn=_setup_pedestrian,
    ),
    # ── Phase 2 : Emergency / advanced ───────────────────────────────────
    Scenario(
        name="emergency_stop",
        description="Sudden obstacle at 8m - emergency braking",
        spawn_idx=13,
        max_steps=200,
        phase=2,
        expected="full stop in under 2 seconds",
        setup_fn=_setup_emergency_stop,
    ),
    Scenario(
        name="lane_change",
        description="Overtake a slow vehicle - lane change",
        spawn_idx=0,  # NOT FOUND in Town02 — set manually
        max_steps=300,
        phase=2,
        expected="change lane and return",
    ),
]
