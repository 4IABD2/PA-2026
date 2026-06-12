"""Launch Phase 1 RL training on CARLA.

Usage:
    python scripts/run_rl_training.py [options]

    --host        CARLA server host (default: localhost)
    --port        CARLA server port (default: 2000)
    --timesteps   total PPO training steps (default: 500 000)
    --tag         short label added to the run folder name
    --demo-eps    number of demo episodes to record after training (default: 3)

Output — one timestamped folder under runs/ :
    runs/YYYY-MM-DD_HH-MM_<tag>/
        params.json              hyperparams + config
        model_best.zip           best checkpoint (saved during training)
        model_final.zip          weights at end of training
        checkpoints/             periodic snapshots (every timesteps/10 steps)
        training_log.csv         per-episode reward / length (Monitor format)
        reward_curve.png         matplotlib reward plot
        progress_XXXXK.mp4       short clips (200 steps) for each checkpoint
        demo.mp4                 full inference with HUD overlay (best model)
        highlights/              one clip per checkpoint on the same fixed route
        demo.mp4                 best model, same fixed route, full episode with HUD

Known limitations (replaced as team modules land):
    - is_on_road = True hardcoded (Karim's semantic seg not yet plugged)
    - CarlaGTDepthEstimator / CarlaGTLaneDetector = GT stubs
    - Victor's Navigation.plan() calls MatplotVisualizer — silenced via Agg backend
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


class _StdoutFilter:
    """Drops single-line debug prints from third-party code during training."""

    _BLOCKED = ("newt command:", "next command:")

    def __init__(self, stream):
        self._out = stream
        self._buf = ""

    def write(self, text: str) -> None:
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if not any(pat in line for pat in self._BLOCKED):
                self._out.write(line + "\n")

    def flush(self) -> None:
        self._out.flush()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")  # must be before any other matplotlib / pyplot import

import carla

from src.dataset.encodings import CAMERA_LOCATION, CAMERA_ROTATION_PITCH
from src.interfaces.navigation_types import HighLevelCommand, Route, Waypoint
from src.interfaces.stubs import CarlaGTDepthEstimator, CarlaGTLaneDetector
from src.navigation.navigation import Navigation
from src.ai.training.rl_env import CarlaEnv
from src.ai.training.rl_train import make_model, train, _PPO_DEFAULTS
from src.ai.training.run_manager import make_run_dir, save_params, plot_reward_curve
from src.ai.inference.rl_demo import load_model, record_episode, eval_model, Scenario


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

    def next_command(self, vehicle_position: Waypoint, route: Route) -> HighLevelCommand:
        try:
            return self._nav.next_command(vehicle_position, route)
        except Exception:
            return HighLevelCommand.LANE_FOLLOW


# ---------------------------------------------------------------------------
# CARLA helpers
# ---------------------------------------------------------------------------

_CAM_W = 200
_CAM_H = 88
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


def _spawn_sensor(world: carla.World, ego: carla.Actor, bp_name: str, **attrs) -> carla.Sensor:
    bp = world.get_blueprint_library().find(bp_name)
    for k, v in attrs.items():
        bp.set_attribute(k, str(v))
    return world.spawn_actor(bp, _CAM_TRANSFORM, attach_to=ego)


def _destroy_all(actors: list) -> None:
    for a in actors:
        if a and a.is_alive:
            a.destroy()


# Fixed seed → reproducible spawn for the final full demo.
_DEMO_RESET_SEED = 42

# Spawn situations for checkpoint comparison clips — Town10HD_Opt.
# Generated by: uv run python3 scripts/explore_spawns.py --host 100.97.91.60
_DEMO_SCENARIOS = [
    Scenario("straight",      spawn_idx=22,  max_steps=500),  # curve=0°
    Scenario("near_junction", spawn_idx=32,  max_steps=500),  # junction=0m
    Scenario("curve",         spawn_idx=62,  max_steps=500),  # curve=42°
    Scenario("traffic_light", spawn_idx=129, max_steps=500),  # tl=24m
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 RL training on CARLA")
    parser.add_argument("--host",              default="localhost")
    parser.add_argument("--port",              type=int, default=2000)
    parser.add_argument("--timesteps",         type=int, default=500_000)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--tag",               default="ppo")
    parser.add_argument("--demo-eps",          type=int, default=3)
    args = parser.parse_args()

    run_dir = make_run_dir(tag=f"{args.tag}_{args.timesteps // 1000}k")
    print(f"Run folder: {run_dir}")

    params = {
        **_PPO_DEFAULTS,
        "timesteps":         args.timesteps,
        "max_episode_steps": args.max_episode_steps,
        "host":              args.host,
        "obs":               "7-scalars-GT",
    }
    save_params(run_dir, params)

    print(f"Connecting to CARLA at {args.host}:{args.port} …")
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    _setup_sync(world)

    ego, sensors = None, []
    try:
        ego          = _spawn_ego(world)
        camera       = _spawn_sensor(world, ego, "sensor.camera.rgb",
                                     image_size_x=_CAM_W, image_size_y=_CAM_H)
        depth_sensor = _spawn_sensor(world, ego, "sensor.camera.depth",
                                     image_size_x=_CAM_W, image_size_y=_CAM_H)
        col_sensor   = _spawn_sensor(world, ego, "sensor.other.collision")
        sensors = [camera, depth_sensor, col_sensor]

        for _ in range(10):          # warm-up: let sensors produce first frames
            world.tick()

        depth_estimator = CarlaGTDepthEstimator(depth_sensor)
        lane_detector   = CarlaGTLaneDetector(world, ego)

        carla_map   = world.get_map()
        nav         = _NavAdapter(Navigation(ego, carla_map))
        spawn_pts   = carla_map.get_spawn_points()
        route       = nav.plan(ego.get_transform().location, spawn_pts[-1].location)

        monitor_base = run_dir / "training_log"   # Monitor appends .monitor.csv itself
        env = CarlaEnv(
            world=world, ego_vehicle=ego, nav=nav, route=route,
            depth_estimator=depth_estimator, lane_detector=lane_detector,
            camera=camera, collision_sensor=col_sensor,
            max_episode_steps=args.max_episode_steps,
        )

        # SB3 Monitor wrapper — logs episode reward/length to CSV automatically
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

        env_monitored = Monitor(env, filename=str(monitor_base))
        model = make_model(env_monitored)

        callbacks = [
            EvalCallback(
                Monitor(env),
                best_model_save_path=str(run_dir),
                log_path=str(run_dir),
                eval_freq=max(args.timesteps // 20, 1000),
                n_eval_episodes=3,
                verbose=0,
            ),
            CheckpointCallback(
                save_freq=max(args.timesteps // 10, 2048),
                save_path=str(run_dir / "checkpoints"),
                name_prefix="rl_model",
                verbose=0,
            ),
        ]

        print(f"Training PPO for {args.timesteps:,} steps …")
        sys.stdout = _StdoutFilter(sys.stdout)
        try:
            model.learn(total_timesteps=args.timesteps, callback=callbacks)
            model.save(str(run_dir / "model_final"))
            print("Training done.")

            # reward curve
            plot_reward_curve(Path(str(monitor_base) + ".monitor.csv"), run_dir / "reward_curve.png")
            print(f"Reward curve → {run_dir / 'reward_curve.png'}")

            # demo + progress videos — swap to high-res camera for all recordings
            if args.demo_eps > 0:
                hud_params = {
                    "lr":    params["learning_rate"],
                    "γ":     params["gamma"],
                    "steps": f"{args.timesteps // 1000}k",
                }

                demo_frame: list = [None]
                demo_cam = _spawn_sensor(
                    world, ego, "sensor.camera.rgb",
                    image_size_x=_DEMO_CAM_W, image_size_y=_DEMO_CAM_H,
                )
                def _on_demo_frame(raw):
                    arr = __import__("numpy").frombuffer(raw.raw_data, dtype="uint8").reshape(raw.height, raw.width, 4)
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

                    ckpt_dir = run_dir / "checkpoints"
                    ckpt_files = sorted(ckpt_dir.glob("rl_model_*_steps.zip"))
                    if ckpt_files:
                        n_select = min(10, max(1, args.timesteps // 10_000))
                        if n_select >= len(ckpt_files):
                            selected = ckpt_files
                        else:
                            step = (len(ckpt_files) - 1) / (n_select - 1)
                            indices = sorted({round(i * step) for i in range(n_select)})
                            selected = [ckpt_files[i] for i in indices]
                        print(f"Running benchmark eval on {len(selected)}/{len(ckpt_files)} checkpoints …")
                        all_results = {}
                        for ckpt in selected:
                            steps_k = int(ckpt.stem.split("_")[2]) // 1000
                            ckpt_model = load_model(str(ckpt))
                            out = str(evals_dir / f"checkpoint_{steps_k:04d}k.mp4")
                            print(f"  checkpoint_{steps_k:04d}k …")
                            ckpt_results = eval_model(
                                ckpt_model, env,
                                output_path=out,
                                fps=20,
                                render_fn=lambda: demo_frame[0],
                            )
                            all_results[f"{steps_k:04d}k"] = ckpt_results
                        print(f"Checkpoint evals → {evals_dir}/checkpoint_*.mp4")

                    # benchmark eval on best model
                    best_model = load_model(str(run_dir / "best_model"))
                    best_results = eval_model(
                        best_model, env,
                        output_path=str(evals_dir / "best_model.mp4"),
                        fps=20,
                        render_fn=lambda: demo_frame[0],
                    )
                    all_results["best_model"] = best_results
                    print(f"Best model eval → {evals_dir / 'best_model.mp4'}")

                    # save all benchmark results to JSON
                    import json
                    results_path = evals_dir / "results.json"
                    with open(results_path, "w") as f:
                        json.dump(all_results, f, indent=2)
                    print(f"Benchmark results → {results_path}")

                    # free-run demo with best model (fixed spawn, no benchmark)
                    record_episode(
                        best_model, env,
                        output_path=str(run_dir / "demo.mp4"),
                        fps=20,
                        hud_params=hud_params,
                        max_steps=args.max_episode_steps,
                        n_episodes=args.demo_eps,
                        render_fn=lambda: demo_frame[0],
                        reset_seed=_DEMO_RESET_SEED,
                    )
                    print(f"Demo video → {run_dir / 'demo.mp4'}")
                finally:
                    demo_cam.stop()
                    demo_cam.destroy()
        finally:
            sys.stdout = sys.stdout._out  # restore

        print(f"\nAll artifacts in: {run_dir}/")

    finally:
        _restore_async(world)
        _destroy_all([ego, *sensors])
        print("Cleanup done.")


if __name__ == "__main__":
    main()
