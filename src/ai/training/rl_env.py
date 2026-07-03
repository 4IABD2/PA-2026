"""CarlaEnv — gymnasium.Env wrapper around CARLA for Phase 1 RL training."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.ai.rewards.reward_fn import compute_reward
from src.interfaces.navigation_types import HighLevelCommand, Route, Waypoint
from src.interfaces.perception_types import ObjectClass

if TYPE_CHECKING:
    import carla
    from src.interfaces.navigation_types import Navigation
    from src.perception.pipeline import PerceptionPipeline

# obs = [speed_norm, cmd_left, cmd_right, cmd_straight,
#         lane_angle_norm, lane_offset_norm, is_on_road,
#         nearest_vehicle_norm, red_light_distance_norm, speed_limit_norm,
#         nearest_walker_norm, nearest_stop_yield_norm]
_OBS_LOW  = np.array([0., 0., 0., 0., -1., -1., 0., 0., 0., 0., 0., 0.], dtype=np.float32)
_OBS_HIGH = np.array([1., 1., 1., 1.,  1.,  1., 1., 1., 1., 1., 1., 1.], dtype=np.float32)

_MAX_SPEED_KMH         = 90.0
_MAX_OBSTACLE_M        = 50.0
_WARMUP_TICKS          = 5     # ticks after teleport so physics settles and sensors fill
_OFF_ROUTE_M           = 15.0  # metres from nearest route waypoint before off_route penalty fires
_ROUTE_GRACE_STEPS     = 20    # steps after reset where off-route is not penalised (car joins route)
_DEFAULT_SPEED_LIMIT_KMH = 50.0

_SPEED_LIMIT_KMH: dict[ObjectClass, float] = {
    ObjectClass.SPEED_30: 30.0,
    ObjectClass.SPEED_40: 40.0,
    ObjectClass.SPEED_60: 60.0,
    ObjectClass.SPEED_90: 90.0,
}

_RED_LIGHT_VIOLATION_DIST_M = 5.0
_RED_LIGHT_VIOLATION_SPEED_KMH = 5.0
_STOP_YIELD_VIOLATION_DIST_M = 5.0
_STOP_YIELD_VIOLATION_SPEED_KMH = 5.0


def _check_violation(
    distance_norm: float, speed_kmh: float, flagged: bool,
    dist_threshold_m: float, speed_threshold_kmh: float,
) -> tuple[bool, bool]:
    """Returns (violation_just_happened, new_flagged_state).

    `flagged` prevents re-penalising every step while the vehicle lingers
    close to the same light/sign; it clears once the vehicle moves away,
    so a later, different light/sign can still trigger a new violation.
    """
    distance_m = distance_norm * _MAX_OBSTACLE_M
    is_close = distance_m < dist_threshold_m
    violation_now = is_close and speed_kmh > speed_threshold_kmh and not flagged
    return violation_now, is_close


def _nearest_distance_norm(objects: list, classes: tuple) -> float:
    """Normalised distance (0-1) to the nearest object of any of `classes`.

    Returns 1.0 (== _MAX_OBSTACLE_M or further away) if none is detected.
    """
    dists = [
        o.distance_m for o in objects
        if o.class_name in classes and o.distance_m is not None
    ]
    return float(np.clip(min(dists) / _MAX_OBSTACLE_M, 0.0, 1.0)) if dists else 1.0


class CarlaEnv(gym.Env):
    def __init__(
        self,
        world: "carla.World",
        ego_vehicle: "carla.Actor",
        nav: "Navigation",
        route: Route,
        perception: "PerceptionPipeline",
        lane_estimate_fn: Callable[[np.ndarray], tuple[str, float, float]],
        camera: "carla.Sensor",
        collision_sensor: "carla.Sensor",
        max_episode_steps: int = 1000,
    ) -> None:
        super().__init__()
        self.world = world
        self.ego = ego_vehicle
        self.nav = nav
        self.route = route
        self.perception = perception
        self._lane_estimate = lane_estimate_fn
        self.max_episode_steps = max_episode_steps

        self.observation_space = spaces.Box(low=_OBS_LOW, high=_OBS_HIGH, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )

        self._collision_flag: bool = False
        self._collision_speed_kmh: float = 0.0
        self._red_light_flagged: bool = False
        self._stop_yield_flagged: bool = False
        self._last_image: np.ndarray | None = None
        self._step_count: int = 0
        self._route_idx: int = 0       # sliding pointer into route.waypoints for efficient off-route check
        self._current_speed_limit_kmh: float = _DEFAULT_SPEED_LIMIT_KMH
        self.off_route_count: int = 0  # steps spent off-route this episode (readable by eval_model)

        collision_sensor.listen(self._on_collision)
        camera.listen(self._on_camera)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        spawn_idx = (options or {}).get("spawn_idx")
        self._teleport_to_spawn(spawn_idx)
        if hasattr(self.nav, 'index_way'):
            self.nav.index_way = 0
        # Replan route from new position so nav commands are correct at every spawn.
        if spawn_idx is not None and hasattr(self.nav, 'plan'):
            try:
                spawn_pts = self.world.get_map().get_spawn_points()
                dest = spawn_pts[-1].location
                self.route = self.nav.plan(self.ego.get_transform().location, dest)
            except Exception:
                pass
        self._collision_flag = False
        self._collision_speed_kmh = 0.0
        self._red_light_flagged = False
        self._stop_yield_flagged = False
        self._step_count = 0
        self._route_idx = 0
        self._current_speed_limit_kmh = _DEFAULT_SPEED_LIMIT_KMH
        self.off_route_count = 0
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
        off_route = self._is_off_route() if self._step_count > _ROUTE_GRACE_STEPS else False
        if off_route:
            self.off_route_count += 1

        red_light_violation, self._red_light_flagged = _check_violation(
            float(obs[8]), speed_kmh, self._red_light_flagged,
            _RED_LIGHT_VIOLATION_DIST_M, _RED_LIGHT_VIOLATION_SPEED_KMH,
        )
        stop_yield_violation, self._stop_yield_flagged = _check_violation(
            float(obs[11]), speed_kmh, self._stop_yield_flagged,
            _STOP_YIELD_VIOLATION_DIST_M, _STOP_YIELD_VIOLATION_SPEED_KMH,
        )

        reward, terminated = compute_reward(
            speed_kmh=speed_kmh,
            center_offset=float(obs[5]),    # lane_offset_norm, from Karim's lane detection
            is_on_road=bool(obs[6] > 0.5),  # from Karim's lane detection
            collision=self._collision_flag,
            off_route=off_route,
            nearest_vehicle_m=float(obs[7]) * _MAX_OBSTACLE_M,
            nearest_walker_m=float(obs[10]) * _MAX_OBSTACLE_M,
            speed_limit_kmh=self._current_speed_limit_kmh,
            collision_speed_kmh=self._collision_speed_kmh,
            red_light_violation=red_light_violation,
            stop_yield_violation=stop_yield_violation,
        )
        truncated = self._step_count >= self.max_episode_steps
        return obs, reward, terminated, truncated, {}

    def _get_obs(self) -> np.ndarray:
        image = self._last_image if self._last_image is not None else np.zeros((720, 1280, 3), dtype=np.uint8)

        # Speed (onboard sensor)
        speed_norm = float(np.clip(self._speed_kmh() / _MAX_SPEED_KMH, 0.0, 1.0))

        # Navigation command from Victor's GPS
        v_transform = self.ego.get_transform()
        v_loc = v_transform.location
        vehicle_wp = Waypoint(x=v_loc.x, y=v_loc.y, z=v_loc.z, yaw_deg=v_transform.rotation.yaw)
        cmd = self.nav.next_command(vehicle_wp, self.route)
        cmd_left     = 1.0 if cmd == HighLevelCommand.LEFT     else 0.0
        cmd_right    = 1.0 if cmd == HighLevelCommand.RIGHT    else 0.0
        cmd_straight = 1.0 if cmd == HighLevelCommand.STRAIGHT else 0.0

        # Karim: lane heading, lateral offset, on-road status
        direction, angle, offset = self._lane_estimate(image)
        lane_angle_norm = float(np.clip(angle / 90.0, -1.0, 1.0))
        lane_offset_norm = float(np.clip(offset, -1.0, 1.0))
        is_on_road = 1.0 if direction != "NONE" else 0.0

        # Franck: nearest vehicle, red light distance, speed limit sign, walker, stop/yield
        objects, _ = self.perception.perceive(image)

        nearest_vehicle_norm = _nearest_distance_norm(objects, (ObjectClass.VEHICLE,))
        red_light_distance_norm = _nearest_distance_norm(objects, (ObjectClass.RED_LIGHT,))
        nearest_walker_norm = _nearest_distance_norm(objects, (ObjectClass.WALKER,))
        nearest_stop_yield_norm = _nearest_distance_norm(objects, (ObjectClass.STOP, ObjectClass.YIELD))

        speed_signs = [o for o in objects if o.class_name in _SPEED_LIMIT_KMH]
        if speed_signs:
            self._current_speed_limit_kmh = _SPEED_LIMIT_KMH[speed_signs[0].class_name]
        speed_limit_norm = float(np.clip(self._current_speed_limit_kmh / _MAX_SPEED_KMH, 0.0, 1.0))

        return np.array(
            [speed_norm, cmd_left, cmd_right, cmd_straight,
             lane_angle_norm, lane_offset_norm, is_on_road,
             nearest_vehicle_norm, red_light_distance_norm, speed_limit_norm,
             nearest_walker_norm, nearest_stop_yield_norm],
            dtype=np.float32,
        )

    def render(self) -> np.ndarray | None:
        return self._last_image

    def _teleport_to_spawn(self, spawn_idx: int | None = None) -> None:
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            return
        if spawn_idx is not None:
            idx = int(spawn_idx) % len(spawn_points)
        else:
            idx = int(self.np_random.integers(len(spawn_points)))
        spawn = spawn_points[idx]
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

    def _is_off_route(self) -> bool:
        """True if ego is more than _OFF_ROUTE_M metres from the nearest route waypoint.

        Uses a sliding window around _route_idx for O(1) amortised cost instead of
        scanning the whole route each step.
        """
        if not (self.route and self.route.waypoints):
            return False
        wps = self.route.waypoints
        n = len(wps)
        start = max(0, self._route_idx - 5)
        end = min(n, self._route_idx + 60)
        ego = self.ego.get_transform().location
        min_dist_sq = float("inf")
        min_idx = self._route_idx
        for i in range(start, end):
            wp = wps[i]
            d_sq = (ego.x - wp.x) ** 2 + (ego.y - wp.y) ** 2
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
                min_idx = i
        self._route_idx = min_idx
        return min_dist_sq > _OFF_ROUTE_M ** 2

    def _on_collision(self, event) -> None:
        self._collision_flag = True
        self._collision_speed_kmh = self._speed_kmh()

    def _on_camera(self, raw_image) -> None:
        arr = np.frombuffer(raw_image.raw_data, dtype=np.uint8).reshape(
            raw_image.height, raw_image.width, 4
        )
        self._last_image = arr[:, :, [2, 1, 0]]  # BGRA → RGB
