"""Run the full benchmark evaluation on a trained PPO model.

Usage:
    uv run python3 scripts/run_eval.py --model runs/<dir>/best_model.zip --host <ip>

Output:
    eval_<model_stem>.mp4   — full 13-scenario video
    Printed per-scenario results in terminal.

Workflow:
    1. Run scripts/explore_spawns.py --host <ip>  to fill missing spawn_idx
    2. Copy spawn_idx values into src/ai/inference/benchmark.py
    3. Run this script with the best model checkpoint
"""

from __future__ import annotations

import argparse
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
from src.ai.inference.rl_demo import load_model, eval_model

# ---------------------------------------------------------------------------
# CARLA helpers (same pattern as run_rl_training.py)
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

    def next_command(
        self, vehicle_position: Waypoint, route: Route
    ) -> HighLevelCommand:
        try:
            return self._nav.next_command(vehicle_position, route)
        except Exception:
            return HighLevelCommand.LANE_FOLLOW


_CAM_W, _CAM_H = 1280, 720
_DEMO_CAM_W, _DEMO_CAM_H = 1280, 720
_CAM_TRANSFORM = carla.Transform(
    carla.Location(x=CAMERA_LOCATION[0], y=CAMERA_LOCATION[1], z=CAMERA_LOCATION[2]),
    carla.Rotation(pitch=CAMERA_ROTATION_PITCH),
)


def _setup_sync(world: carla.World, fps: int = 20) -> None:
    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = 1.0 / fps
    world.apply_settings(s)


def _restore_async(world: carla.World) -> None:
    s = world.get_settings()
    s.synchronous_mode = False
    world.apply_settings(s)


def _spawn_sensor(
    world: carla.World, ego: carla.Actor, bp_name: str, transform=None, **attrs
) -> carla.Sensor:
    bp = world.get_blueprint_library().find(bp_name)
    for k, v in attrs.items():
        bp.set_attribute(k, str(v))
    t = transform or _CAM_TRANSFORM
    return world.spawn_actor(bp, t, attach_to=ego)


def _destroy_all(actors: list) -> None:
    for a in actors:
        try:
            if a and a.is_alive:
                a.destroy()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark evaluation on CARLA")
    parser.add_argument("--model", required=True, help="Path to .zip model file")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument(
        "--ground-truth-lane",
        action="store_true",
        help="feed lane offset / on-road from CARLA map geometry instead of the "
        "detector — must match how the model was trained (e.g. v15)",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    output_path = str(model_path.parent / f"eval_{model_path.stem}.mp4")
    print(f"Model  : {model_path}")
    print(f"Output : {output_path}")

    print(f"Connecting to CARLA at {args.host}:{args.port} …")
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    _setup_sync(world)

    ego, sensors = None, []
    try:
        bp = world.get_blueprint_library().find("vehicle.tesla.model3")
        ego = world.spawn_actor(bp, world.get_map().get_spawn_points()[0])

        camera = _spawn_sensor(
            world, ego, "sensor.camera.rgb", image_size_x=_CAM_W, image_size_y=_CAM_H
        )
        col_sensor = _spawn_sensor(world, ego, "sensor.other.collision")
        sensors = [camera, col_sensor]

        for _ in range(10):
            world.tick()

        print("Loading perception models …")
        perception = PerceptionPipeline()

        carla_map = world.get_map()
        nav = _NavAdapter(Navigation(ego, carla_map))
        spawn_pts = carla_map.get_spawn_points()
        route = nav.plan(ego.get_transform().location, spawn_pts[-1].location)

        env = CarlaEnv(
            world=world,
            ego_vehicle=ego,
            nav=nav,
            route=route,
            perception=perception,
            lane_estimate_fn=lane_estimate,
            camera=camera,
            collision_sensor=col_sensor,
            max_episode_steps=500,
            use_ground_truth_lane=args.ground_truth_lane,
        )

        # high-res demo camera
        demo_frame: list = [None]
        demo_cam = _spawn_sensor(
            world,
            ego,
            "sensor.camera.rgb",
            image_size_x=_DEMO_CAM_W,
            image_size_y=_DEMO_CAM_H,
        )

        def _on_frame(raw):
            arr = np.frombuffer(raw.raw_data, dtype="uint8").reshape(
                raw.height, raw.width, 4
            )
            demo_frame[0] = arr[:, :, [2, 1, 0]]

        demo_cam.listen(_on_frame)
        for _ in range(5):
            world.tick()

        model = load_model(str(model_path))
        print(f"\nRunning 13-scenario benchmark …\n")

        try:
            results = eval_model(
                model,
                env,
                output_path=output_path,
                fps=20,
                render_fn=lambda: demo_frame[0],
            )
        finally:
            demo_cam.stop()
            demo_cam.destroy()

        # ── save JSON ────────────────────────────────────────────────────
        import json

        results_path = str(model_path.parent / f"eval_{model_path.stem}.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # ── print summary ────────────────────────────────────────────────
        print("\n" + "=" * 55)
        print("BENCHMARK RESULTS")
        print("=" * 55)
        n_p1 = sum(1 for r in results.values() if r["success"] is not None)
        n_ok = sum(1 for r in results.values() if r["success"] is True)
        for name, r in results.items():
            s = r["success"]
            mark = "✓" if s is True else ("✗" if s is False else "–")
            detail = (
                f"steps={r['steps']:<4}  "
                f"dist={r['max_dist_from_start']:5.1f}m  "
                f"crash={str(r['terminated']):<5}  "
                f"offset={r['center_offset']['mean_abs']:.3f}  "
                f"speed={r['speed']['mean']:.1f}km/h  "
                f"off_route={r['off_route_pct']:.0%}"
            )
            print(f"  {mark} {name:<22} {detail}")
        print(f"\nPhase 1 : {n_ok}/{n_p1} passed")
        print(f"Video   : {output_path}")
        print(f"Results : {results_path}")

    finally:
        _restore_async(world)
        _destroy_all([ego, *sensors])
        print("Cleanup done.")


if __name__ == "__main__":
    main()
