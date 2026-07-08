"""Analyse les spawn points de la map CARLA et sélectionne les meilleurs par catégorie.

Usage:
    uv run python3 scripts/explore_spawns.py --host <carla-ip> [--port 2000]

Sortie : indices de spawn pour chaque catégorie de scénario du benchmark.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import carla

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _dist(a: carla.Location, b: carla.Location) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def _yaw_change(
    wp: carla.Waypoint, ahead_m: float = 30.0, step_m: float = 2.0
) -> float:
    """Total absolute yaw variation over the next ahead_m metres — high = curve."""
    cur = wp
    yaws = [cur.transform.rotation.yaw]
    dist = 0.0
    while dist < ahead_m:
        nexts = cur.next(step_m)
        if not nexts:
            break
        cur = nexts[0]
        yaws.append(cur.transform.rotation.yaw)
        dist += step_m
    if len(yaws) < 2:
        return 0.0
    diffs = [abs(((b - a + 180) % 360) - 180) for a, b in zip(yaws, yaws[1:])]
    return sum(diffs)


def _yaw_change_signed(
    wp: carla.Waypoint, ahead_m: float = 30.0, step_m: float = 2.0
) -> float:
    """Net yaw change over next ahead_m metres. Positive = right turn, negative = left turn."""
    cur = wp
    yaw_start = cur.transform.rotation.yaw
    dist = 0.0
    while dist < ahead_m:
        nexts = cur.next(step_m)
        if not nexts:
            break
        cur = nexts[0]
        dist += step_m
    return ((cur.transform.rotation.yaw - yaw_start + 180) % 360) - 180


def _junction_turn_dirs(
    wp: carla.Waypoint, ahead_m: float = 25.0, step_m: float = 2.0
) -> set[str]:
    """Returns available turn directions {'left', 'right', 'straight'} at the next junction.

    Walks ahead_m to find the junction, then checks all exit branches.
    """
    cur = wp
    dist = 0.0
    yaw_in = wp.transform.rotation.yaw

    while dist < ahead_m:
        nexts = cur.next(step_m)
        if not nexts:
            return set()
        if nexts[0].is_junction:
            break
        cur = nexts[0]
        dist += step_m

    if dist >= ahead_m and not (cur.next(step_m) or [False])[0]:
        return set()

    yaw_in = cur.transform.rotation.yaw
    exits = cur.next(step_m)
    dirs: set[str] = set()
    for exit_wp in exits:
        # walk a bit through the junction to see the exit direction
        probe = exit_wp
        for _ in range(4):
            nxt = probe.next(step_m)
            if nxt:
                probe = nxt[0]
        delta = ((probe.transform.rotation.yaw - yaw_in + 180) % 360) - 180
        if abs(delta) < 30:
            dirs.add("straight")
        elif delta > 30:
            dirs.add("right")
        elif delta < -30:
            dirs.add("left")
    return dirs


def _nearest_traffic_light(loc: carla.Location, lights) -> float:
    if not lights:
        return float("inf")
    return min(_dist(loc, tl.get_location()) for tl in lights)


def _nearest_speed_limit(
    loc: carla.Location, speed_limit_actors, radius: float = 30.0
) -> int | None:
    """Returns speed limit in km/h if a sign is within radius, else None.

    Speed limit actors have type_id like 'traffic.speed_limit.30'.
    """
    best_dist = float("inf")
    best_limit = None
    for sl in speed_limit_actors:
        d = _dist(loc, sl.get_location())
        if d < radius and d < best_dist:
            best_dist = d
            try:
                best_limit = int(sl.type_id.split(".")[-1])
            except (ValueError, IndexError):
                pass
    return best_limit


def _has_lane_change(wp: carla.Waypoint) -> bool:
    return int(wp.lane_change) != 0  # carla.LaneChange.NONE == 0


# ---------------------------------------------------------------------------
# Spawn classifier
# ---------------------------------------------------------------------------


@dataclass
class SpawnInfo:
    idx: int
    loc: carla.Location
    yaw_deg: float
    road_id: int
    is_junction: bool
    near_junction: bool
    junction_dist_m: float
    tl_dist_m: float
    curve_deg: float
    curve_dir: str = ""  # 'left', 'right', or 'straight'
    junction_dirs: set = field(default_factory=set)
    speed_limit: int | None = None
    lane_change: bool = False
    tags: list[str] = field(default_factory=list)

    def classify(self) -> None:
        # ── existing tags ────────────────────────────────────────────────
        if self.curve_deg < 15 and not self.near_junction:
            self.tags.append("straight")
        if self.near_junction and self.junction_dist_m < 30:
            self.tags.append("near_junction")
        if self.curve_deg > 30:
            self.tags.append("curve")
        if self.tl_dist_m < 40:
            self.tags.append("traffic_light")

        # ── new tags ─────────────────────────────────────────────────────
        if self.curve_deg > 20 and self.curve_dir == "left":
            self.tags.append("curve_left")
        if self.curve_deg > 20 and self.curve_dir == "right":
            self.tags.append("curve_right")
        if "left" in self.junction_dirs and self.junction_dist_m < 40:
            self.tags.append("turn_left")
        if "right" in self.junction_dirs and self.junction_dist_m < 40:
            self.tags.append("turn_right")
        if self.junction_dirs == {"straight"} and self.junction_dist_m < 40:
            self.tags.append("junction_straight")
        if self.lane_change and not self.near_junction:
            self.tags.append("multi_lane")
        if self.speed_limit is not None and self.speed_limit <= 30:
            self.tags.append("speed_zone")


def _analyse_spawns(world: carla.World) -> list[SpawnInfo]:
    cmap = world.get_map()
    spawn_pts = cmap.get_spawn_points()
    lights = list(world.get_actors().filter("traffic.traffic_light"))
    speed_limits = list(world.get_actors().filter("traffic.speed_limit.*"))

    infos = []
    for idx, sp in enumerate(spawn_pts):
        loc = sp.location
        wp = cmap.get_waypoint(
            loc, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if wp is None:
            continue

        # junction distance
        cur, dist, jdist = wp, 0.0, float("inf")
        while dist < 60.0:
            if cur.is_junction:
                jdist = dist
                break
            nexts = cur.next(2.0)
            if not nexts:
                break
            cur = nexts[0]
            dist += 2.0

        curve_deg = _yaw_change(wp)
        signed = _yaw_change_signed(wp)
        if abs(signed) < 10:
            curve_dir = "straight"
        elif signed > 0:
            curve_dir = "right"
        else:
            curve_dir = "left"

        junction_dirs = _junction_turn_dirs(wp) if jdist < 40.0 else set()

        info = SpawnInfo(
            idx=idx,
            loc=loc,
            yaw_deg=sp.rotation.yaw,
            road_id=wp.road_id,
            is_junction=wp.is_junction,
            near_junction=jdist < 60.0,
            junction_dist_m=jdist,
            tl_dist_m=_nearest_traffic_light(loc, lights),
            curve_deg=curve_deg,
            curve_dir=curve_dir,
            junction_dirs=junction_dirs,
            speed_limit=_nearest_speed_limit(loc, speed_limits),
            lane_change=_has_lane_change(wp),
        )
        info.classify()
        infos.append(info)

    return infos


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def _best(
    infos: list[SpawnInfo], tag: str, exclude: set[int], n: int = 1
) -> list[SpawnInfo]:
    candidates = [i for i in infos if tag in i.tags and i.idx not in exclude]
    candidates.sort(key=lambda i: (len(i.tags), i.junction_dist_m))
    chosen = candidates[:n]
    exclude.update(c.idx for c in chosen)
    return chosen


def _pick_all(infos: list[SpawnInfo]) -> dict[str, SpawnInfo | None]:
    """Pick one spawn per benchmark scenario category."""
    exclude: set[int] = set()
    result: dict[str, SpawnInfo | None] = {}

    # Phase 1 — pick in dependency order to avoid reusing good spawns
    for tag in (
        "straight",
        "curve_left",
        "curve_right",
        "turn_left",
        "turn_right",
        "junction_straight",
        "traffic_light",
        "speed_zone",
        "multi_lane",
    ):
        chosen = _best(infos, tag, exclude, n=1)
        result[tag] = chosen[0] if chosen else None

    # NPC scenarios reuse base spawns (no exclusion)
    result["npc_follow"] = result.get("straight")
    result["npc_crossing"] = result.get("turn_left") or result.get("near_junction")
    result["red_light"] = result.get("traffic_light")
    result["pedestrian"] = None  # needs manual identification (crosswalk geometry)
    result["emergency_stop"] = result.get("straight")
    result["lane_change"] = result.get("multi_lane")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args()

    print(f"Connecting to CARLA at {args.host}:{args.port} …")
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    print(f"Map: {world.get_map().name}")
    print("Analysing spawn points …")
    infos = _analyse_spawns(world)
    print(f"  {len(infos)} spawn points analysed.\n")

    picks = _pick_all(infos)

    # ── per-category summary ──────────────────────────────────────────────
    print("=" * 65)
    print("Spawn indices for benchmark scenarios")
    print("=" * 65)

    # Ordered list matching BENCHMARK_SCENARIOS
    scenario_order = [
        ("straight", "Phase 1 — route droite"),
        ("curve_left", "Phase 1 — virage gauche"),
        ("curve_right", "Phase 1 — virage droit"),
        ("turn_left", "Phase 1 — carrefour gauche"),
        ("turn_right", "Phase 1 — carrefour droit"),
        ("junction_straight", "Phase 1 — carrefour tout droit"),
        ("npc_follow", "Phase 1 — NPC lent devant  (=straight)"),
        ("npc_crossing", "Phase 1 — NPC qui coupe   (=turn_left)"),
        ("red_light", "Phase 2 — feu rouge        (=traffic_light)"),
        ("speed_zone", "Phase 2 — zone 30 km/h"),
        ("pedestrian", "Phase 2 — piéton (manual)"),
        ("emergency_stop", "Phase 2 — frein urgence    (=straight)"),
        ("lane_change", "Phase 2 — dépassement      (=multi_lane)"),
    ]

    for sc_name, label in scenario_order:
        sp = picks.get(sc_name)
        if sp is None:
            idx_str = "NOT FOUND"
            detail = "—"
        else:
            idx_str = str(sp.idx)
            tl = f"{sp.tl_dist_m:.0f}m" if sp.tl_dist_m < 999 else "—"
            jd = f"{sp.junction_dist_m:.0f}m" if sp.junction_dist_m < 999 else "—"
            sl = f"{sp.speed_limit}" if sp.speed_limit else "—"
            detail = (
                f"curve={sp.curve_deg:.0f}°{sp.curve_dir[0].upper()}  "
                f"junc={jd}  tl={tl}  limit={sl}  "
                f"dirs={sorted(sp.junction_dirs) or '—'}"
            )
        print(f"  {sc_name:<22} idx={idx_str:<5} {label}")
        if sp:
            print(f"  {'':22}     ({detail})")

    # ── ready-to-paste spawn_idx table ────────────────────────────────────
    print("\n" + "=" * 65)
    print("Copy these spawn_idx values into src/ai/inference/benchmark.py")
    print("=" * 65)
    for sc_name, _ in scenario_order:
        sp = picks.get(sc_name)
        idx = sp.idx if sp else 0
        note = ""
        if sp is None:
            note = "  # NOT FOUND — set manually"
        elif sc_name in (
            "npc_follow",
            "npc_crossing",
            "red_light",
            "emergency_stop",
            "lane_change",
        ):
            note = "  # reuses above"
        print(f"  {sc_name:<22} spawn_idx={idx},{note}")


if __name__ == "__main__":
    main()
