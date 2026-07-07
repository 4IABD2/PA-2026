"""Find dest_spawn_idx for junction scenarios using CARLA's waypoint graph.

Walks forward from each scenario spawn to find the nearest junction, then
follows each branch to identify which spawn points lie in each direction
(left / right / straight). No call to Victor's nav planner — pure CARLA API.

Usage:
    uv run python3 scripts/find_dest_spawns.py --host 100.97.91.60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib; matplotlib.use("Agg")

import carla


PROBE_SCENARIOS = [
    ("turn_left",         79),
    ("turn_right",        81),
    ("junction_straight", 47),
]

# How far (metres) to walk forward looking for a junction.
LOOK_AHEAD_M = 120
STEP_M = 2.0

# How far to walk along each junction branch before snapping to nearest spawn.
BRANCH_WALK_M = 40
BRANCH_STEP_M = 3.0

# Min distance from ego to a candidate destination (avoid trivially nearby spawns).
MIN_DEST_DIST_M = 30.0


def _yaw_delta(yaw_a: float, yaw_b: float) -> float:
    """Signed angle from yaw_b to yaw_a, in [-180, 180]."""
    return ((yaw_a - yaw_b + 180) % 360) - 180


def find_junction_branches(
    carla_map: carla.Map,
    spawn_pts: list,
    spawn_idx: int,
) -> dict[str, list[tuple[int, float]]]:
    """
    From spawn_idx, walk forward until a junction is found, then follow each
    branch.  Returns {direction: [(dest_spawn_idx, dist_m), ...]} where
    direction ∈ {"left", "right", "straight"}.
    """
    origin = spawn_pts[spawn_idx]
    ego_yaw = origin.rotation.yaw

    wp = carla_map.get_waypoint(
        origin.location, project_to_road=True, lane_type=carla.LaneType.Driving
    )
    if wp is None:
        return {}

    # Walk forward to find junction entry
    walked = 0.0
    while walked < LOOK_AHEAD_M:
        nexts = wp.next(STEP_M)
        if not nexts:
            break
        wp = nexts[0]
        walked += STEP_M
        if wp.is_junction:
            break

    if not wp.is_junction:
        return {"info": f"no junction found within {LOOK_AHEAD_M} m"}

    # Collect exit waypoints of the junction
    try:
        junction = wp.get_junction()
        pairs = junction.get_waypoints(carla.LaneType.Driving)
        exit_wps = [end for (_start, end) in pairs]
    except Exception:
        exit_wps = []

    # Also include all branches from wp.next() at junction
    for branch in wp.next(STEP_M):
        exit_wps.append(branch)

    branches: dict[str, list[tuple[int, float]]] = {}

    for exit_wp in exit_wps:
        # Walk further along this branch
        cur = exit_wp
        for _ in range(int(BRANCH_WALK_M / BRANCH_STEP_M)):
            nexts = cur.next(BRANCH_STEP_M)
            if not nexts:
                break
            cur = nexts[0]

        branch_loc = cur.transform.location
        branch_yaw = cur.transform.rotation.yaw
        delta = _yaw_delta(branch_yaw, ego_yaw)

        if delta < -40:
            direction = "left"
        elif delta > 40:
            direction = "right"
        else:
            direction = "straight"

        # Find nearest spawn point along this branch
        best_idx, best_dist = -1, float("inf")
        for i, sp in enumerate(spawn_pts):
            if i == spawn_idx:
                continue
            d_from_ego = sp.location.distance(origin.location)
            if d_from_ego < MIN_DEST_DIST_M:
                continue
            d = sp.location.distance(branch_loc)
            if d < best_dist:
                best_dist = d
                best_idx = i

        if best_idx >= 0:
            lst = branches.setdefault(direction, [])
            # Avoid duplicates
            if not any(idx == best_idx for idx, _ in lst):
                lst.append((best_idx, best_dist))

    return branches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()
    spawn_pts = carla_map.get_spawn_points()

    print(f"\nAnalysing junction branches for {len(PROBE_SCENARIOS)} scenarios …")
    print("=" * 60)

    for sc_name, sc_spawn in PROBE_SCENARIOS:
        print(f"\n[{sc_name}]  ego=spawn_{sc_spawn}")
        branches = find_junction_branches(carla_map, spawn_pts, sc_spawn)

        if isinstance(branches.get("info"), str):
            print(f"  {branches['info']}")
            continue
        if not branches:
            print("  ✗ no branches found — check spawn_idx")
            continue

        for direction, candidates in sorted(branches.items()):
            marker = "→" if sc_name == f"turn_{direction}" or \
                     (sc_name == "junction_straight" and direction == "straight") \
                     else " "
            for dest_idx, snap_dist in candidates:
                dist_from_ego = spawn_pts[dest_idx].location.distance(
                    spawn_pts[sc_spawn].location
                )
                print(f"  {marker} {direction:8s}  dest_spawn_idx={dest_idx:4d}"
                      f"   {dist_from_ego:6.1f} m from ego"
                      f"   (snap dist {snap_dist:.1f} m)")

        # Print recommended value for this scenario
        desired = ("left"     if "turn_left"  in sc_name else
                   "right"    if "turn_right" in sc_name else
                   "straight")
        candidates = branches.get(desired, [])
        if candidates:
            rec = candidates[0][0]
            print(f"\n  → paste into benchmark.py:  dest_spawn_idx={rec}")
        else:
            print(f"\n  ✗ no '{desired}' branch found — try a different spawn_idx")

    print("\n" + "=" * 60)
    print("Done — no nav planner called, pure waypoint graph.\n")


if __name__ == "__main__":
    main()
