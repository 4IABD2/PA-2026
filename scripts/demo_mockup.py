"""Demo mockup — two rule-based policies on the 4 scenarios, no training needed.

Policies
--------
straight     : steer=0, full throttle → crashes at intersections / NPCs
route_follow : P-controller on center_offset → stays on road

Usage:
    uv run python3 scripts/demo_mockup.py --host <carla-ip> [--npcs 15]

Output (current directory):
    mockup_straight.mp4
    mockup_route_follow.mp4
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import carla

from src.dataset.encodings import CAMERA_LOCATION, CAMERA_ROTATION_PITCH
from src.interfaces.navigation_types import HighLevelCommand, Route, Waypoint
from src.navigation.navigation import Navigation
from src.perception.pipeline import PerceptionPipeline
from src.lane_detection.lane_perception import estimate as lane_estimate
from src.ai.training.rl_env import CarlaEnv
from src.ai.inference.rl_demo import record_episode, Scenario


# ---------------------------------------------------------------------------
# Same 4 scenarios as run_rl_training.py
# ---------------------------------------------------------------------------

DEMO_SCENARIOS = [
    Scenario("straight",      spawn_idx=22,  max_steps=500),
    Scenario("near_junction", spawn_idx=32,  max_steps=500),
    Scenario("curve",         spawn_idx=62,  max_steps=500),
    Scenario("traffic_light", spawn_idx=129, max_steps=500),
]

_CAM_W, _CAM_H = 1280, 720
_DEMO_CAM_W, _DEMO_CAM_H = 1280, 720
_CAM_TRANSFORM = carla.Transform(
    carla.Location(x=CAMERA_LOCATION[0], y=CAMERA_LOCATION[1], z=CAMERA_LOCATION[2]),
    carla.Rotation(pitch=CAMERA_ROTATION_PITCH),
)


# ---------------------------------------------------------------------------
# Mock policies — no training, pure rules
# ---------------------------------------------------------------------------

class StraightPolicy:
    """Full throttle, zero steer — will crash at intersections and NPCs."""

    name = "straight"

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        action = np.array([0.0, 0.9, 0.0], dtype=np.float32)
        return action, None


class RouteFollowPolicy:
    """P-controller on lane_offset_norm — roughly follows the road.

    obs[4] = lane_offset_norm (lateral deviation from lane centre)
    obs[0] = speed_norm       (speed / 90 km/h)
    """

    name = "route_follow"

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        lane_offset_norm = float(obs[4])
        speed_norm       = float(obs[0])

        steer    = float(np.clip(-lane_offset_norm * 0.25, -1.0, 1.0))
        throttle = 0.45 if speed_norm < 0.25 else 0.3
        brake    = 0.0
        return np.array([steer, throttle, brake], dtype=np.float32), None


# ---------------------------------------------------------------------------
# CARLA helpers
# ---------------------------------------------------------------------------

class _NavAdapter:
    def __init__(self, nav: Navigation) -> None:
        self._nav = nav

    @property
    def index_way(self) -> int:
        return self._nav.index_way

    @index_way.setter
    def index_way(self, v: int) -> None:
        self._nav.index_way = v

    def plan(self, start, destination) -> Route:
        self._nav.index_way = 0
        return self._nav.plan(start, destination)

    def next_command(self, vehicle_position: Waypoint, route: Route) -> HighLevelCommand:
        try:
            return self._nav.next_command(vehicle_position, route)
        except Exception:
            return HighLevelCommand.LANE_FOLLOW


def _setup_sync(world: carla.World, fps: int = 20) -> None:
    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = 1.0 / fps
    world.apply_settings(s)


def _restore_async(world: carla.World, tm: carla.TrafficManager | None) -> None:
    s = world.get_settings()
    s.synchronous_mode = False
    world.apply_settings(s)
    if tm is not None:
        tm.set_synchronous_mode(False)


def _spawn_sensor(world, ego, bp_name, transform=None, **attrs):
    bp = world.get_blueprint_library().find(bp_name)
    for k, v in attrs.items():
        bp.set_attribute(k, str(v))
    t = transform or _CAM_TRANSFORM
    return world.spawn_actor(bp, t, attach_to=ego)


def _spawn_npcs(world: carla.World, client: carla.Client, n: int,
                exclude_spawns: set[int]) -> list[carla.Actor]:
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)
    bp_lib = world.get_blueprint_library()
    vehicle_bps = list(bp_lib.filter("vehicle.*"))
    spawn_pts = world.get_map().get_spawn_points()
    npcs = []
    indices = [i for i in range(len(spawn_pts)) if i not in exclude_spawns]
    random.shuffle(indices)
    for i in indices[:n]:
        bp = random.choice(vehicle_bps)
        npc = world.try_spawn_actor(bp, spawn_pts[i])
        if npc:
            npc.set_autopilot(True, tm.get_port())
            npcs.append(npc)
    return npcs


def _destroy_all(actors: list) -> None:
    for a in actors:
        if a and a.is_alive:
            a.destroy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--npcs", type=int, default=15,
                        help="NPC vehicles spawned for collision scenarios")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    tm = client.get_trafficmanager()
    _setup_sync(world, fps=20)

    ego, sensors, npcs = None, [], []
    try:
        bp = world.get_blueprint_library().find("vehicle.tesla.model3")
        ego = world.spawn_actor(bp, world.get_map().get_spawn_points()[0])

        camera     = _spawn_sensor(world, ego, "sensor.camera.rgb",
                                   image_size_x=_CAM_W, image_size_y=_CAM_H)
        col_sensor = _spawn_sensor(world, ego, "sensor.other.collision")
        sensors = [camera, col_sensor]

        for _ in range(10):
            world.tick()

        print("Loading perception models …")
        perception = PerceptionPipeline()

        # spawn NPCs — exclude the 4 scenario spawn indices
        scenario_spawns = {sc.spawn_idx for sc in DEMO_SCENARIOS}
        npcs = _spawn_npcs(world, client, args.npcs, exclude_spawns=scenario_spawns)
        print(f"Spawned {len(npcs)} NPC vehicles.")
        for _ in range(20):   # let NPCs settle
            world.tick()

        carla_map = world.get_map()
        nav = _NavAdapter(Navigation(ego, carla_map))
        spawn_pts = carla_map.get_spawn_points()
        route = nav.plan(ego.get_transform().location, spawn_pts[-1].location)

        env = CarlaEnv(
            world=world, ego_vehicle=ego, nav=nav, route=route,
            perception=perception, lane_estimate_fn=lane_estimate,
            camera=camera, collision_sensor=col_sensor,
            max_episode_steps=500,
        )

        # high-res demo camera
        demo_frame: list = [None]
        demo_cam = _spawn_sensor(
            world, ego, "sensor.camera.rgb",
            image_size_x=_DEMO_CAM_W, image_size_y=_DEMO_CAM_H,
        )
        def _on_demo_frame(raw):
            arr = np.frombuffer(raw.raw_data, dtype="uint8").reshape(
                raw.height, raw.width, 4)
            demo_frame[0] = arr[:, :, [2, 1, 0]]
        demo_cam.listen(_on_demo_frame)
        for _ in range(5):
            world.tick()

        try:
            for policy in [StraightPolicy(), RouteFollowPolicy()]:
                out = f"mockup_{policy.name}.mp4"
                print(f"\nRecording {out} …")
                record_episode(
                    policy, env,
                    output_path=out,
                    fps=20,
                    render_fn=lambda: demo_frame[0],
                    scenarios=DEMO_SCENARIOS,
                )
                print(f"  → {out}")
        finally:
            demo_cam.stop()
            demo_cam.destroy()

        print("\nDone. Check mockup_straight.mp4 and mockup_route_follow.mp4")

    finally:
        _restore_async(world, tm)
        _destroy_all(npcs)
        _destroy_all([ego, *sensors])
        print("Cleanup done.")


if __name__ == "__main__":
    main()
