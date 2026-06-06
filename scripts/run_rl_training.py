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
        params.json        hyperparams + config
        model_best.zip     best checkpoint (saved during training)
        model_final.zip    weights at end of training
        training_log.csv   per-episode reward / length (Monitor format)
        reward_curve.png   matplotlib reward plot
        demo.mp4           recorded inference with HUD overlay

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

from src.interfaces.navigation_types import HighLevelCommand, Route, Waypoint
from src.interfaces.stubs import CarlaGTDepthEstimator, CarlaGTLaneDetector
from src.navigation.navigation import Navigation
from src.ai.training.rl_env import CarlaEnv
from src.ai.training.rl_train import make_model, train, _PPO_DEFAULTS
from src.ai.training.run_manager import make_run_dir, save_params, plot_reward_curve
from src.ai.inference.rl_demo import load_model, record_episode


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
_CAM_TRANSFORM = carla.Transform(carla.Location(x=0.30, y=0.0, z=1.50))


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
        ]

        print(f"Training PPO for {args.timesteps:,} steps …")
        sys.stdout = _StdoutFilter(sys.stdout)
        try:
            model.learn(total_timesteps=args.timesteps, callback=callbacks)
        finally:
            sys.stdout = sys.stdout._out  # restore
        model.save(str(run_dir / "model_final"))
        print("Training done.")

        # reward curve
        plot_reward_curve(Path(str(monitor_base) + ".monitor.csv"), run_dir / "reward_curve.png")
        print(f"Reward curve → {run_dir / 'reward_curve.png'}")

        # demo video
        if args.demo_eps > 0:
            hud_params = {
                "lr":    params["learning_rate"],
                "γ":     params["gamma"],
                "steps": f"{args.timesteps // 1000}k",
            }
            demo_path = str(run_dir / "demo.mp4")
            record_episode(
                model, env,
                output_path=demo_path,
                fps=20,
                hud_params=hud_params,
                max_steps=args.max_episode_steps,
                n_episodes=args.demo_eps,
            )
            print(f"Demo video → {demo_path}")

        print(f"\nAll artifacts in: {run_dir}/")

    finally:
        _restore_async(world)
        _destroy_all([ego, *sensors])
        print("Cleanup done.")


if __name__ == "__main__":
    main()
