"""CarlaEnv — gymnasium.Env wrapper around CARLA for Phase 1 RL training."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.ai.rewards.reward_fn import compute_reward
from src.interfaces.navigation_types import HighLevelCommand, Route, Waypoint
from src.interfaces.perception_types import DepthEstimator, LaneDetector

if TYPE_CHECKING:
    import carla
    from src.interfaces.navigation_types import Navigation

# obs = [speed_norm, cmd_left, cmd_right, cmd_straight, center_offset, obstacle_norm, heading_norm]
_OBS_LOW  = np.array([0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0], dtype=np.float32)
_OBS_HIGH = np.array([1.0, 1.0, 1.0, 1.0,  1.0, 1.0,  1.0], dtype=np.float32)

_MAX_SPEED_KMH = 90.0
_MAX_OBSTACLE_M = 50.0
_WARMUP_TICKS = 5  # ticks after teleport so physics settles and sensors fill


class CarlaEnv(gym.Env):
    def __init__(
        self,
        world: "carla.World",
        ego_vehicle: "carla.Actor",
        nav: "Navigation",
        route: Route,
        depth_estimator: DepthEstimator,
        lane_detector: LaneDetector,
        camera: "carla.Sensor",
        collision_sensor: "carla.Sensor",
        max_episode_steps: int = 1000,
    ) -> None:
        super().__init__()
        self.world = world
        self.ego = ego_vehicle
        self.nav = nav
        self.route = route
        self.depth_estimator = depth_estimator
        self.lane_detector = lane_detector
        self.max_episode_steps = max_episode_steps

        self.observation_space = spaces.Box(low=_OBS_LOW, high=_OBS_HIGH, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )

        self._collision_flag: bool = False
        self._last_image: np.ndarray | None = None
        self._step_count: int = 0

        collision_sensor.listen(lambda _: setattr(self, "_collision_flag", True))
        camera.listen(self._on_camera)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._teleport_to_spawn()
        if hasattr(self.nav, 'index_way'):
            self.nav.index_way = 0
        self._collision_flag = False
        self._step_count = 0
        self._last_image = None
        for _ in range(_WARMUP_TICKS):
            self.world.tick()
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        steer = float(action[0])
        throttle = float(action[1])
        brake = float(action[2])
        self._apply_control(steer, throttle, brake)
        self.world.tick()
        self._step_count += 1

        obs = self._get_obs()
        speed_kmh = self._speed_kmh()
        # is_on_road: replaced by semantic segmentation when Karim's model is ready
        reward, terminated = compute_reward(
            speed_kmh=speed_kmh,
            center_offset=float(obs[4]),
            is_on_road=True,
            collision=self._collision_flag,
        )
        truncated = self._step_count >= self.max_episode_steps
        return obs, reward, terminated, truncated, {}

    def _get_obs(self) -> np.ndarray:
        v_transform = self.ego.get_transform()
        v_loc = v_transform.location
        wp = self.world.get_map().get_waypoint(v_loc, project_to_road=True)

        speed_norm = np.clip(self._speed_kmh() / _MAX_SPEED_KMH, 0.0, 1.0)

        vehicle_wp = Waypoint(x=v_loc.x, y=v_loc.y, z=v_loc.z, yaw_deg=v_transform.rotation.yaw)
        cmd = self.nav.next_command(vehicle_wp, self.route)
        cmd_left    = 1.0 if cmd == HighLevelCommand.LEFT     else 0.0
        cmd_right   = 1.0 if cmd == HighLevelCommand.RIGHT    else 0.0
        cmd_straight = 1.0 if cmd == HighLevelCommand.STRAIGHT else 0.0

        image = self._last_image if self._last_image is not None else np.zeros((88, 200, 3), dtype=np.uint8)
        lanes = self.lane_detector.detect(image)
        center_offset = float(lanes.center_offset) if lanes.center_offset is not None else 0.0

        try:
            depth = self.depth_estimator.estimate(image)
            obstacle_norm = float(np.clip(depth.min() / _MAX_OBSTACLE_M, 0.0, 1.0))
        except RuntimeError:
            obstacle_norm = 1.0  # assume clear road if no depth frame yet

        heading_error = ((v_transform.rotation.yaw - wp.transform.rotation.yaw + 180) % 360) - 180
        heading_norm = float(np.clip(heading_error / 180.0, -1.0, 1.0))

        return np.array(
            [speed_norm, cmd_left, cmd_right, cmd_straight, center_offset, obstacle_norm, heading_norm],
            dtype=np.float32,
        )

    def render(self) -> np.ndarray | None:
        return self._last_image

    def _teleport_to_spawn(self) -> None:
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            return
        spawn = spawn_points[int(self.np_random.integers(len(spawn_points)))]
        self.ego.set_transform(spawn)
        try:
            from carla import Vector3D  # noqa: PLC0415
            self.ego.set_target_velocity(Vector3D(0, 0, 0))
        except ModuleNotFoundError:
            self.ego.set_target_velocity(None)

    def _apply_control(self, steer: float, throttle: float, brake: float) -> None:
        try:
            from carla import VehicleControl  # noqa: PLC0415
            control = VehicleControl(throttle=throttle, steer=steer, brake=brake)
        except ModuleNotFoundError:
            control = None  # tests: ego is a Mock, accepts any argument
        self.ego.apply_control(control)

    def _speed_kmh(self) -> float:
        v = self.ego.get_velocity()
        return math.sqrt(v.x**2 + v.y**2 + v.z**2) * 3.6

    def _on_camera(self, raw_image) -> None:
        arr = np.frombuffer(raw_image.raw_data, dtype=np.uint8).reshape(
            raw_image.height, raw_image.width, 4
        )
        self._last_image = arr[:, :, [2, 1, 0]]  # BGRA → RGB
