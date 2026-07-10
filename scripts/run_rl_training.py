"""Launch Phase 1 RL training on CARLA.

Usage:
    python scripts/run_rl_training.py [options]

    --host        CARLA server host (default: localhost)
    --port        CARLA server port (default: 2000)
    --timesteps   total PPO training steps (default: 500 000)
    --tag         short label added to the run folder name
    --demo-eps    number of demo episodes to record after training (default: 3)
    --yolo-weights  path to YOLO best.pt (default: src/perception/yolo/weights/best.pt)
    --npcs          number of NPC vehicles spawned (default: 18)
    --pedestrians   number of NPC pedestrians spawned (default: 6)

Output — one timestamped folder under runs/ :
    runs/YYYY-MM-DD_HH-MM_<tag>/
        params.json              hyperparams + config
        best_model.zip           best checkpoint (saved during training)
        model_final.zip          weights at end of training
        checkpoints/             periodic snapshots (every timesteps/10 steps)
        training_log.csv         per-episode reward / length (Monitor format)
        reward_curve.png         matplotlib reward plot
        demo.mp4                 full inference with HUD overlay (best model)
        evals/                   benchmark eval per checkpoint + best model

Obs (11 scalars): speed | cmd_left | cmd_right | cmd_straight |
                  lane_offset | is_on_road | nearest_vehicle | red_light_distance |
                  speed_limit | nearest_walker | nearest_stop_yield
Perception: Franck's PerceptionPipeline (YOLO11s + Depth Anything v2) + Karim's YOLOPv2 lane detection.
Environment: NPC vehicles (autopilot) + NPC pedestrians (AI walker controllers).
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path


def _format_duration(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


class _Tee:
    """Duplicates every line to a real stream and a log file.

    Buffers partial writes and only forwards complete lines, so line-based
    filtering (dropping noisy third-party prints) works correctly. The log
    file is flushed after every line — a crash must not lose the last output.
    """

    _BLOCKED = ("newt command:", "next command:")

    def __init__(self, stream, log_file, filter_lines: bool = False):
        self._out = stream
        self._log = log_file
        self._filter = filter_lines
        self._buf = ""

    def write(self, text: str) -> None:
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if self._filter and any(pat in line for pat in self._BLOCKED):
                continue
            self._out.write(line + "\n")
            self._log.write(line + "\n")
            self._log.flush()

    def flush(self) -> None:
        self._out.flush()
        self._log.flush()


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # must be before any other matplotlib / pyplot import

import carla

from src.dataset.encodings import CAMERA_LOCATION, CAMERA_ROTATION_PITCH
from src.interfaces.navigation_types import HighLevelCommand, Route, Waypoint
from src.navigation.navigation import Navigation
from src.perception.pipeline import PerceptionPipeline
from src.lane_detection.lane_perception import estimate as lane_estimate
from src.ai.training.rl_env import (
    CarlaEnv,
    _OBS_LOW,
    _OFF_ROUTE_M,
    _MAX_OBSTACLE_M,
    _WARMUP_TICKS,
    _ROUTE_GRACE_STEPS,
    _DEFAULT_SPEED_LIMIT_KMH,
    _DEST_REACHED_RADIUS_M,
    _MIN_TRAVEL_FOR_DEST_M,
)
from src.ai.training.rl_train import make_model, train, _PPO_DEFAULTS
from src.ai.training.run_manager import make_run_dir, save_params, plot_reward_curve
from src.ai.inference.rl_demo import (
    load_model,
    record_episode,
    eval_model,
    pick_best_checkpoint,
    Scenario,
)
from src.ai.inference.benchmark import _MIN_DIST_M
from src.ai.rewards.reward_fn import (
    MAX_SPEED_KMH,
    _W_SPEED,
    _W_CENTER,
    _W_ALIVE,
    _P_OFFROAD,
    _P_COLLISION_BASE,
    _P_COLLISION_SPEED_SCALE,
    _P_STALL,
    _P_OFF_ROUTE,
    _W_FOLLOWING,
    _SAFE_HEADWAY_S,
    _W_WALKER_PROXIMITY,
    _WALKER_DANGER_M,
    _W_SPEEDING,
    _SPEEDING_TOLERANCE_KMH,
    _P_RED_LIGHT_VIOLATION,
    _P_STOP_YIELD_VIOLATION,
    _P_DEST_REACHED,
    _W_SAFE_DRIVING,
    _W_JERK,
    REWARD_COMPONENT_KEYS,
)

# ---------------------------------------------------------------------------
# Nav adapter
# ---------------------------------------------------------------------------


class _NavAdapter:
    """Guards Victor's Navigation against IndexError at end of route."""

    def __init__(self, nav: Navigation) -> None:
        self._nav = nav

    @property
    def index_way(self) -> int:
        return self._nav.index_way

    @index_way.setter
    def index_way(self, v: int) -> None:
        self._nav.index_way = v

    def plan(self, start: carla.Location, destination: carla.Location) -> Route:
        self._nav.index_way = 0
        return self._nav.plan(start, destination)

    def next_command(
        self, vehicle_position: Waypoint, route: Route
    ) -> HighLevelCommand:
        try:
            return self._nav.next_command(vehicle_position, route)
        except Exception:
            return HighLevelCommand.LANE_FOLLOW


# ---------------------------------------------------------------------------
# CARLA helpers
# ---------------------------------------------------------------------------

_CAM_W = 1280
_CAM_H = 720
_CAM_TRANSFORM = carla.Transform(
    carla.Location(x=CAMERA_LOCATION[0], y=CAMERA_LOCATION[1], z=CAMERA_LOCATION[2]),
    carla.Rotation(pitch=CAMERA_ROTATION_PITCH),
)

_DEMO_CAM_W = 1280
_DEMO_CAM_H = 720


def _setup_sync(world: carla.World, fps: int = 20) -> None:
    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = 1.0 / fps
    world.apply_settings(s)


def _restore_async(world: carla.World) -> None:
    s = world.get_settings()
    s.synchronous_mode = False
    world.apply_settings(s)


def _spawn_ego(world: carla.World) -> carla.Vehicle:
    bp = world.get_blueprint_library().find("vehicle.tesla.model3")
    return world.spawn_actor(bp, world.get_map().get_spawn_points()[0])


def _spawn_sensor(
    world: carla.World, ego: carla.Actor, bp_name: str, **attrs
) -> carla.Sensor:
    bp = world.get_blueprint_library().find(bp_name)
    for k, v in attrs.items():
        bp.set_attribute(k, str(v))
    return world.spawn_actor(bp, _CAM_TRANSFORM, attach_to=ego)


def _destroy_all(actors: list) -> None:
    for a in actors:
        if a and a.is_alive:
            a.destroy()


def _spawn_npc_vehicles(
    world: carla.World, client: carla.Client, n: int, exclude_spawn_idx: int
) -> list:
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)
    bp_lib = world.get_blueprint_library()
    vehicle_bps = list(bp_lib.filter("vehicle.*"))
    spawn_pts = world.get_map().get_spawn_points()
    indices = [i for i in range(len(spawn_pts)) if i != exclude_spawn_idx]
    random.shuffle(indices)
    npcs = []
    for i in indices[:n]:
        bp = random.choice(vehicle_bps)
        npc = world.try_spawn_actor(bp, spawn_pts[i])
        if npc:
            npc.set_autopilot(True, tm.get_port())
            npcs.append(npc)
    return npcs


def _spawn_npc_pedestrians(world: carla.World, n: int) -> list:
    bp_lib = world.get_blueprint_library()
    walker_bps = list(bp_lib.filter("walker.pedestrian.*"))
    ctrl_bp = bp_lib.find("controller.ai.walker")
    actors = []
    for _ in range(n):
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        bp = random.choice(walker_bps)
        walker = world.try_spawn_actor(bp, carla.Transform(loc))
        if walker is None:
            continue
        world.tick()
        ctrl = world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=walker)
        ctrl.start()
        dest = world.get_random_location_from_navigation()
        if dest is not None:
            ctrl.go_to_location(dest)
        ctrl.set_max_speed(1.4)
        actors.extend([ctrl, walker])
    return actors


def _destroy_pedestrians(actors: list) -> None:
    for a in actors:
        if not a or not a.is_alive:
            continue
        if "controller" in a.type_id:
            a.stop()
        a.destroy()


# Fixed seed → reproducible spawn for the final full demo.
_DEMO_RESET_SEED = 42

# Spawn situations for checkpoint comparison clips — Town10HD_Opt. Unused
# (superseded by BENCHMARK_SCENARIOS in src/ai/inference/benchmark.py) — kept
# here for reference, not recalibrated for Town02.
# Generated by: uv run python3 scripts/explore_spawns.py --host 100.97.91.60
_DEMO_SCENARIOS = [
    Scenario("straight", spawn_idx=22, max_steps=500),  # curve=0°
    Scenario("near_junction", spawn_idx=32, max_steps=500),  # junction=0m
    Scenario("curve", spawn_idx=62, max_steps=500),  # curve=42°
    Scenario("traffic_light", spawn_idx=129, max_steps=500),  # tl=24m
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 RL training on CARLA")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--tag", default="ppo")
    parser.add_argument("--demo-eps", type=int, default=3)
    parser.add_argument("--yolo-weights", default="src/perception/yolo/weights/best.pt")
    parser.add_argument(
        "--depth-model",
        default="depth-anything/Depth-Anything-V2-Small-hf",
        help="HuggingFace model ID or local path to the depth model",
    )
    parser.add_argument(
        "--npcs",
        type=int,
        default=18,
        help="number of NPC vehicles spawned in autopilot",
    )
    parser.add_argument(
        "--pedestrians", type=int, default=6, help="number of NPC pedestrians spawned"
    )
    args = parser.parse_args()

    run_dir = make_run_dir(tag=f"{args.tag}_{args.timesteps // 1000}k")
    print(f"Run folder: {run_dir}")

    log_file = open(run_dir / "run.log", "w", encoding="utf-8")
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_stdout, log_file, filter_lines=True)
    sys.stderr = _Tee(orig_stderr, log_file, filter_lines=False)
    try:
        params = {
            **_PPO_DEFAULTS,
            "timesteps": args.timesteps,
            "max_episode_steps": args.max_episode_steps,
            "host": args.host,
            "obs": f"{len(_OBS_LOW)}-scalars-traffic",
            "reward": {
                "max_speed_kmh": MAX_SPEED_KMH,
                "w_speed": _W_SPEED,
                "w_center": _W_CENTER,
                "w_alive": _W_ALIVE,
                "p_offroad": _P_OFFROAD,
                "p_collision_base": _P_COLLISION_BASE,
                "p_collision_speed_scale": _P_COLLISION_SPEED_SCALE,
                "p_stall": _P_STALL,
                "p_off_route": _P_OFF_ROUTE,
                "w_following": _W_FOLLOWING,
                "safe_headway_s": _SAFE_HEADWAY_S,
                "w_walker_proximity": _W_WALKER_PROXIMITY,
                "walker_danger_m": _WALKER_DANGER_M,
                "w_speeding": _W_SPEEDING,
                "speeding_tolerance_kmh": _SPEEDING_TOLERANCE_KMH,
                "p_red_light_violation": _P_RED_LIGHT_VIOLATION,
                "p_stop_yield_violation": _P_STOP_YIELD_VIOLATION,
                "p_dest_reached": _P_DEST_REACHED,
                "w_safe_driving": _W_SAFE_DRIVING,
                "w_jerk": _W_JERK,
            },
            "env": {
                "off_route_m": _OFF_ROUTE_M,
                "route_grace_steps": _ROUTE_GRACE_STEPS,
                "max_obstacle_m": _MAX_OBSTACLE_M,
                "warmup_ticks": _WARMUP_TICKS,
                "min_dist_m": _MIN_DIST_M,
                "default_speed_limit_kmh": _DEFAULT_SPEED_LIMIT_KMH,
                "dest_reached_radius_m": _DEST_REACHED_RADIUS_M,
                "min_travel_for_dest_m": _MIN_TRAVEL_FOR_DEST_M,
                "npcs": args.npcs,
                "pedestrians": args.pedestrians,
            },
        }
        save_params(run_dir, params)

        print(f"Connecting to CARLA at {args.host}:{args.port} …")
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        world = client.get_world()
        _setup_sync(world)

        ego, sensors, npc_vehicles, npc_walkers = None, [], [], []
        try:
            ego = _spawn_ego(world)
            camera = _spawn_sensor(
                world,
                ego,
                "sensor.camera.rgb",
                image_size_x=_CAM_W,
                image_size_y=_CAM_H,
            )
            col_sensor = _spawn_sensor(world, ego, "sensor.other.collision")
            sensors = [camera, col_sensor]

            for _ in range(10):  # warm-up: let sensors produce first frames
                world.tick()

            print(
                f"Spawning {args.npcs} NPC vehicles + {args.pedestrians} pedestrians …"
            )
            npc_vehicles = _spawn_npc_vehicles(
                world, client, args.npcs, exclude_spawn_idx=0
            )
            npc_walkers = _spawn_npc_pedestrians(world, args.pedestrians)
            for _ in range(20):  # let NPCs settle before training starts
                world.tick()
            print(
                f"Spawned {len(npc_vehicles)} vehicles, {len(npc_walkers) // 2} pedestrians."
            )

            print("Loading perception models (YOLO + Depth Anything + YOLOPv2) …")
            perception = PerceptionPipeline(
                yolo_weights=args.yolo_weights,
                depth_model_name=args.depth_model,
            )
            # lane_estimate is a module-level singleton — loaded on first call

            carla_map = world.get_map()
            nav = _NavAdapter(Navigation(ego, carla_map))
            spawn_pts = carla_map.get_spawn_points()
            route = nav.plan(ego.get_transform().location, spawn_pts[-1].location)

            monitor_base = (
                run_dir / "training_log"
            )  # Monitor appends .monitor.csv itself
            env = CarlaEnv(
                world=world,
                ego_vehicle=ego,
                nav=nav,
                route=route,
                perception=perception,
                lane_estimate_fn=lane_estimate,
                camera=camera,
                collision_sensor=col_sensor,
                max_episode_steps=args.max_episode_steps,
            )

            # SB3 Monitor wrapper — logs episode reward/length to CSV automatically
            from stable_baselines3.common.monitor import Monitor
            from stable_baselines3.common.callbacks import CheckpointCallback

            env_monitored = Monitor(
                env,
                filename=str(monitor_base),
                info_keywords=REWARD_COMPONENT_KEYS,
            )
            model = make_model(env_monitored)

            callbacks = [
                CheckpointCallback(
                    save_freq=max(args.timesteps // 10, 2048),
                    save_path=str(run_dir / "checkpoints"),
                    name_prefix="rl_model",
                    verbose=0,
                ),
            ]

            print(f"Training PPO for {args.timesteps:,} steps …")
            train_start = time.time()
            model.learn(total_timesteps=args.timesteps, callback=callbacks)
            train_duration = time.time() - train_start
            model.save(str(run_dir / "model_final"))
            print("Training done.")

            # reward curve
            plot_reward_curve(
                Path(str(monitor_base) + ".monitor.csv"), run_dir / "reward_curve.png"
            )
            print(f"Reward curve → {run_dir / 'reward_curve.png'}")

            # demo + progress videos — swap to high-res camera for all recordings
            if args.demo_eps > 0:
                hud_params = {
                    "lr": params["learning_rate"],
                    "γ": params["gamma"],
                    "steps": f"{args.timesteps // 1000}k",
                }

                demo_frame: list = [None]
                demo_cam = _spawn_sensor(
                    world,
                    ego,
                    "sensor.camera.rgb",
                    image_size_x=_DEMO_CAM_W,
                    image_size_y=_DEMO_CAM_H,
                )

                def _on_demo_frame(raw):
                    arr = (
                        __import__("numpy")
                        .frombuffer(raw.raw_data, dtype="uint8")
                        .reshape(raw.height, raw.width, 4)
                    )
                    demo_frame[0] = arr[:, :, [2, 1, 0]]  # BGRA → RGB

                demo_cam.listen(_on_demo_frame)
                for _ in range(5):
                    world.tick()

                try:
                    evals_dir = run_dir / "evals"
                    evals_dir.mkdir(exist_ok=True)

                    # full benchmark eval on up to 10 evenly-spaced checkpoints
                    # spacing = 1 eval per 10k steps → 50k=5, 100k=10, 500k=10
                    all_results: dict = {}
                    checkpoint_paths: dict = {}

                    ckpt_dir = run_dir / "checkpoints"
                    ckpt_files = sorted(ckpt_dir.glob("rl_model_*_steps.zip"))
                    if ckpt_files:
                        n_select = min(10, max(1, args.timesteps // 10_000))
                        if n_select <= 1:
                            # A single slot was requested (short/smoke runs) — the last
                            # checkpoint is the most representative of trained performance.
                            selected = [ckpt_files[-1]]
                        elif n_select >= len(ckpt_files):
                            selected = ckpt_files
                        else:
                            step = (len(ckpt_files) - 1) / (n_select - 1)
                            indices = sorted({round(i * step) for i in range(n_select)})
                            selected = [ckpt_files[i] for i in indices]
                        print(
                            f"Running benchmark eval on {len(selected)}/{len(ckpt_files)} checkpoints …"
                        )
                        all_results = {}
                        for ckpt in selected:
                            steps_k = int(ckpt.stem.split("_")[2]) // 1000
                            ckpt_model = load_model(str(ckpt))
                            out = str(evals_dir / f"checkpoint_{steps_k:04d}k.mp4")
                            print(f"  checkpoint_{steps_k:04d}k …")
                            ckpt_results = eval_model(
                                ckpt_model,
                                env,
                                output_path=out,
                                fps=20,
                                render_fn=lambda: demo_frame[0],
                            )
                            label = f"{steps_k:04d}k"
                            all_results[label] = ckpt_results
                            checkpoint_paths[label] = ckpt
                        print(f"Checkpoint evals → {evals_dir}/checkpoint_*.mp4")

                    # save all benchmark results to JSON
                    import json

                    results_path = evals_dir / "results.json"
                    with open(results_path, "w") as f:
                        json.dump(all_results, f, indent=2)
                    print(f"Benchmark results → {results_path}")

                    # Pick the real winner from the full benchmark above (raw training
                    # reward, which an EvalCallback-style pick would rely on, was shown
                    # not to track real quality -- see JOURNAL.md, ppo_v6.3_300k).
                    if not all_results:
                        raise RuntimeError(
                            "No checkpoints were saved during training -- nothing to "
                            "benchmark or pick as best_model. Check CheckpointCallback's "
                            "save_freq against --timesteps."
                        )
                    winner_label = pick_best_checkpoint(all_results)
                    winner_model = load_model(str(checkpoint_paths[winner_label]))
                    winner_model.save(str(run_dir / "best_model"))
                    winner_scenarios = all_results[winner_label]
                    winner_successes = sum(
                        1 for m in winner_scenarios.values() if m.get("success") is True
                    )
                    winner_off_route = sum(
                        m["off_route_pct"] for m in winner_scenarios.values()
                    ) / len(winner_scenarios)
                    print(
                        f"Best model (by benchmark): {winner_label} "
                        f"(successes={winner_successes}, off_route_pct={winner_off_route:.1%})"
                    )

                    # free-run demo with the real best model (fixed spawn, no benchmark)
                    record_episode(
                        winner_model,
                        env,
                        output_path=str(run_dir / "demo.mp4"),
                        fps=20,
                        hud_params=hud_params,
                        max_steps=args.max_episode_steps,
                        n_episodes=args.demo_eps,
                        render_fn=lambda: demo_frame[0],
                        reset_seed=_DEMO_RESET_SEED,
                        spawn_idx=0,
                    )
                    print(f"Demo video → {run_dir / 'demo.mp4'}")
                finally:
                    demo_cam.stop()
                    demo_cam.destroy()

            print(f"\nAll artifacts in: {run_dir}/")
            print(f"Training time: {_format_duration(train_duration)}")

        finally:
            _restore_async(world)
            _destroy_all([ego, *sensors, *npc_vehicles])
            _destroy_pedestrians(npc_walkers)
            print("Cleanup done.")
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        log_file.close()


if __name__ == "__main__":
    main()
