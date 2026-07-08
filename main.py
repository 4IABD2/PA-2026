import argparse
import sys

import carla
import numpy as np
from ai.inference.rl_demo import load_model
from ai.training.rl_env import CarlaEnv
from perception.pipeline import PerceptionPipeline

sys.path.insert(0, "web/client/build_py")
import client_bindings

from scripts.demo_mockup import (
    _setup_sync,
    _spawn_sensor,
    _CAM_W,
    _CAM_H,
    _NavAdapter,
    _DEMO_CAM_W,
    _DEMO_CAM_H,
)
from src.lane_detection.lane_perception import estimate as lane_estimate
from src.navigation.navigation import Navigation


class Main:

    def __init__(self):
        parser = argparse.ArgumentParser(description="Benchmark evaluation on CARLA")
        parser.add_argument("--host", default="localhost")
        parser.add_argument("--port", type=int, default=2000)
        args = parser.parse_args()

        print(f"Connecting to CARLA at {args.host}:{args.port} …")
        client = carla.Client(args.host, args.port)
        world = client.get_world()
        _setup_sync(world)

        bp = world.get_blueprint_library().find("vehicle.tesla.model3")
        ego = world.spawn_actor(bp, world.get_map().get_spawn_points()[0])

        camera = _spawn_sensor(
            world, ego, "sensor.camera.rgb", image_size_x=_CAM_W, image_size_y=_CAM_H
        )
        col_sensor = _spawn_sensor(world, ego, "sensor.other.collision")
        self.sensors = [camera, col_sensor]

        for _ in range(10):
            world.tick()

        print("Loading perception models …")
        perception = PerceptionPipeline()

        carla_map = world.get_map()
        nav = _NavAdapter(Navigation(ego, carla_map))
        spawn_pts = carla_map.get_spawn_points()
        route = nav.plan(ego.get_transform().location, spawn_pts[-1].location)

        self.env = CarlaEnv(
            world=world,
            ego_vehicle=ego,
            nav=nav,
            route=route,
            perception=perception,
            lane_estimate_fn=lane_estimate,
            camera=camera,
            collision_sensor=col_sensor,
            max_episode_steps=500,
        )

        self.demo_frame = [None]
        self.demo_cam = _spawn_sensor(
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
            self.demo_frame[0] = arr[:, :, [2, 1, 0]]

        self.demo_cam.listen(_on_frame)

        for _ in range(5):
            world.tick()

    def run(self):
        action = [1.0, 0.0, 0.0]
        while True:
            if action[-1] < 0.0:
                action[-1] = 0.0
            obs, reward, terminated, truncated, _ = self.env.step(action)
            print(obs)
            action = client_bindings.launch_request(obs)
            print(action)

            try:
                spectator = self.env.world.get_spectator()
                cam_transform = self.demo_cam.get_transform()
                spectator.set_transform(cam_transform)
            except Exception as e:
                print("Could not set spectator transform:", e)


if __name__ == "__main__":
    main = Main()
    main.run()
