"""CarlaEnv — gymnasium.Env wrapper around CARLA for Phase 1 RL training."""

from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.ai.rewards.reward_fn import compute_reward, REWARD_COMPONENT_KEYS
from src.interfaces.navigation_types import HighLevelCommand, Route, Waypoint
from src.interfaces.perception_types import ObjectClass

if TYPE_CHECKING:
    import carla
    from src.interfaces.navigation_types import Navigation
    from src.perception.pipeline import PerceptionPipeline

_OBS_LOW = np.array(
    [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0],
    dtype=np.float32,
)
_OBS_HIGH = np.array(
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    dtype=np.float32,
)

_MAX_SPEED_KMH = 90.0
_MAX_OBSTACLE_M = 50.0
_WARMUP_TICKS = 5
_OFF_ROUTE_M = 15.0
_PROGRESS_GAMMA = 0.99
_ROUTE_LOOKAHEAD_WPS = 3
_ROUTE_GRACE_STEPS = 20
_DEFAULT_SPEED_LIMIT_KMH = 30.0

_SPAWN_SAFETY_MIN_DIST_M = 10.0
_SPAWN_SAFETY_MAX_ATTEMPTS = 10

_DEST_PICK_MAX_ATTEMPTS = 10

_DEST_REACHED_RADIUS_M = 15.0
_MIN_TRAVEL_FOR_DEST_M = 12.0

_CURRICULUM_MIN_FLOOR_M = 18.0
_CURRICULUM_START_MAX_M = 30.0
_CURRICULUM_STEP_M = 8.0
_CURRICULUM_CEILING_M = 70.0
_CURRICULUM_WINDOW = 20
_CURRICULUM_ADVANCE_RATE = 0.5

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
    distance_norm: float,
    speed_kmh: float,
    flagged: bool,
    dist_threshold_m: float,
    speed_threshold_kmh: float,
) -> tuple[bool, bool]:
    """Returns (violation_just_happened, new_flagged_state)."""
    distance_m = distance_norm * _MAX_OBSTACLE_M
    is_close = distance_m < dist_threshold_m
    violation_now = is_close and speed_kmh > speed_threshold_kmh and not flagged
    return violation_now, is_close


def _nearest_distance_norm(objects: list, classes: tuple) -> float:
    """Normalised distance (0-1) to the nearest object of any of `classes`."""
    dists = [
        o.distance_m
        for o in objects
        if o.class_name in classes and o.distance_m is not None
    ]
    return float(np.clip(min(dists) / _MAX_OBSTACLE_M, 0.0, 1.0)) if dists else 1.0


def _in_ego_path(obj, image_width: float) -> bool:
    """True if obj is roughly ahead of the ego (bbox in the middle third)."""
    cx = (obj.bbox[0] + obj.bbox[2]) / 2.0
    return image_width / 3.0 <= cx <= 2.0 * image_width / 3.0


def _signed_lane_offset_norm(
    ego_x: float,
    ego_y: float,
    wp_x: float,
    wp_y: float,
    right_x: float,
    right_y: float,
    lane_width: float,
) -> float:
    """Signed lateral offset of the ego from lane centre, normalised to [-1, 1]."""
    half_width = max(lane_width / 2.0, 0.1)
    lateral = (ego_x - wp_x) * right_x + (ego_y - wp_y) * right_y
    return float(np.clip(lateral / half_width, -1.0, 1.0))


def _signed_bearing_norm(
    ego_x: float, ego_y: float, yaw_deg: float, dest_x: float, dest_y: float
) -> float:
    """Signed heading error from ego to destination, normalised to [-1, 1]."""
    bearing = math.atan2(dest_y - ego_y, dest_x - ego_x)
    heading = math.radians(yaw_deg)
    err = math.atan2(math.sin(bearing - heading), math.cos(bearing - heading))
    return float(np.clip(err / math.pi, -1.0, 1.0))


class CarlaEnv(gym.Env):
    def __init__(
        self,
        world: "carla.World",
        ego_vehicle: "carla.Actor",
        nav: "Navigation",
        route: Route,
        perception: "PerceptionPipeline",
        lane_estimate_fn: Callable[
            [np.ndarray], tuple[str, float, float, float | None]
        ],
        camera: "carla.Sensor",
        collision_sensor: "carla.Sensor",
        max_episode_steps: int = 1000,
        use_ground_truth_lane: bool = False,
        use_goal_bearing: bool = False,
        goal_bearing_lookahead_wps: int = _ROUTE_LOOKAHEAD_WPS,
    ) -> None:
        super().__init__()
        self.world = world
        self.ego = ego_vehicle
        self.nav = nav
        self._use_ground_truth_lane = use_ground_truth_lane
        self._use_goal_bearing = use_goal_bearing
        self._goal_bearing_lookahead_wps = goal_bearing_lookahead_wps
        self.route = route
        self.perception = perception
        self._lane_estimate = lane_estimate_fn
        self.max_episode_steps = max_episode_steps

        obs_low, obs_high = _OBS_LOW, _OBS_HIGH
        if use_goal_bearing:
            obs_low = np.append(obs_low, -1.0).astype(np.float32)
            obs_high = np.append(obs_high, 1.0).astype(np.float32)
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

        self._collision_flag: bool = False
        self._collision_speed_kmh: float = 0.0
        self._red_light_flagged: bool = False
        self._stop_yield_flagged: bool = False
        self._episode_reward_components: dict[str, float] = {
            k: 0.0 for k in REWARD_COMPONENT_KEYS
        }
        self._episode_start_location: "carla.Location | None" = None
        self._prev_steer: float = 0.0
        self._prev_accel: float = 0.0
        self._dist_to_dest_initial: float = 1.0
        self._prev_dist_to_dest_norm: float = 1.0
        self._route_cum: list | None = None
        self._route_total: float | None = None
        self._prev_route_progress_norm: float = 1.0
        self._last_lane_offset_norm: float = 0.0
        self._last_is_on_road: float = 1.0
        self._last_image: np.ndarray | None = None
        self._step_count: int = 0
        self._stall_steps: int = 0
        self._curriculum_max_m: float = _CURRICULUM_START_MAX_M
        self._recent_success: deque = deque(maxlen=_CURRICULUM_WINDOW)
        self._route_idx: int = 0
        self._current_speed_limit_kmh: float = _DEFAULT_SPEED_LIMIT_KMH
        self.off_route_count: int = 0
        self.last_objects: list = []

        collision_sensor.listen(self._on_collision)
        camera.listen(self._on_camera)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        spawn_idx = (options or {}).get("spawn_idx")
        self._teleport_to_spawn(spawn_idx)
        if hasattr(self.nav, "index_way"):
            self.nav.index_way = 0
        for _ in range(_WARMUP_TICKS):
            self.world.tick()
        self._maybe_advance_curriculum()
        if hasattr(self.nav, "plan"):
            try:
                spawn_pts = self.world.get_map().get_spawn_points()
                dest = self._pick_random_destination(
                    spawn_pts, self.ego.get_transform().location
                )
                self.route = self.nav.plan(self.ego.get_transform().location, dest)
            except Exception as exc:
                print(f"    nav.plan failed at reset: {exc}")
        dist0 = self._dist_to_destination()
        self._dist_to_dest_initial = max(dist0, 1.0) if dist0 is not None else 1.0
        self._prev_dist_to_dest_norm = (
            dist0 / self._dist_to_dest_initial if dist0 is not None else 1.0
        )
        self._route_idx = 0
        self._compute_route_cumulative()
        self._prev_route_progress_norm = (
            self._route_remaining_dist() / self._route_total
            if self._route_total
            else 1.0
        )
        self._collision_flag = False
        self._collision_speed_kmh = 0.0
        self._red_light_flagged = False
        self._stop_yield_flagged = False
        self._episode_reward_components = {k: 0.0 for k in REWARD_COMPONENT_KEYS}
        self._episode_start_location = self.ego.get_transform().location
        self._prev_steer = 0.0
        self._prev_accel = 0.0
        self._last_lane_offset_norm = 0.0
        self._last_is_on_road = 1.0
        self._step_count = 0
        self._stall_steps = 0
        self._current_speed_limit_kmh = _DEFAULT_SPEED_LIMIT_KMH
        self.off_route_count = 0
        self._last_image = None
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        steer = float(action[0])
        accel = float(action[1])
        throttle = max(accel, 0.0)
        brake = max(-accel, 0.0)
        self._apply_control(steer, throttle, brake)
        self.world.tick()
        self._step_count += 1

        steer_delta = abs(steer - self._prev_steer)
        self._prev_steer = steer
        self._prev_accel = accel

        obs = self._get_obs()
        speed_kmh = self._speed_kmh()
        off_route = (
            self._is_off_route() if self._step_count > _ROUTE_GRACE_STEPS else False
        )
        if off_route:
            self.off_route_count += 1

        red_light_violation, self._red_light_flagged = _check_violation(
            float(obs[7]),
            speed_kmh,
            self._red_light_flagged,
            _RED_LIGHT_VIOLATION_DIST_M,
            _RED_LIGHT_VIOLATION_SPEED_KMH,
        )
        stop_yield_violation, self._stop_yield_flagged = _check_violation(
            float(obs[10]),
            speed_kmh,
            self._stop_yield_flagged,
            _STOP_YIELD_VIOLATION_DIST_M,
            _STOP_YIELD_VIOLATION_SPEED_KMH,
        )

        dist_to_dest = self._dist_to_destination()
        dist_norm = (
            dist_to_dest / self._dist_to_dest_initial
            if dist_to_dest is not None
            else 1.0
        )

        if self._route_total is not None:
            rem_norm = self._route_remaining_dist() / self._route_total
            progress_delta = _PROGRESS_GAMMA * (-rem_norm) - (
                -self._prev_route_progress_norm
            )
            self._prev_route_progress_norm = rem_norm
        elif dist_to_dest is not None:
            progress_delta = _PROGRESS_GAMMA * (-dist_norm) - (
                -self._prev_dist_to_dest_norm
            )
            self._prev_dist_to_dest_norm = dist_norm
        else:
            progress_delta = 0.0

        reward, terminated, components = compute_reward(
            speed_kmh=speed_kmh,
            center_offset=float(obs[4]),
            is_on_road=bool(obs[5] > 0.5),
            collision=self._collision_flag,
            off_route=off_route,
            nearest_vehicle_m=float(obs[6]) * _MAX_OBSTACLE_M,
            nearest_walker_m=float(obs[9]) * _MAX_OBSTACLE_M,
            red_light_distance_m=float(obs[7]) * _MAX_OBSTACLE_M,
            speed_limit_kmh=self._current_speed_limit_kmh,
            collision_speed_kmh=self._collision_speed_kmh,
            red_light_violation=red_light_violation,
            stop_yield_violation=stop_yield_violation,
            reached_destination=self._reached_destination(),
            steer_delta=steer_delta,
            progress_delta=progress_delta,
            remaining_frac=dist_norm,
            stall_steps=self._stall_steps,
        )
        self._stall_steps = self._stall_steps + 1 if components["r_stall"] < 0.0 else 0
        for key, value in components.items():
            self._episode_reward_components[key] += value

        truncated = self._step_count >= self.max_episode_steps
        if terminated or truncated:
            self._recent_success.append(components["r_destination"] > 0.0)
        info = (
            dict(self._episode_reward_components) if (terminated or truncated) else {}
        )
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        image = (
            self._last_image
            if self._last_image is not None
            else np.zeros((720, 1280, 3), dtype=np.uint8)
        )

        speed_norm = float(np.clip(self._speed_kmh() / _MAX_SPEED_KMH, 0.0, 1.0))

        v_transform = self.ego.get_transform()
        v_loc = v_transform.location
        vehicle_wp = Waypoint(
            x=v_loc.x, y=v_loc.y, z=v_loc.z, yaw_deg=v_transform.rotation.yaw
        )
        cmd = self.nav.next_command(vehicle_wp, self.route)
        cmd_left = 1.0 if cmd == HighLevelCommand.LEFT else 0.0
        cmd_right = 1.0 if cmd == HighLevelCommand.RIGHT else 0.0
        cmd_straight = 1.0 if cmd == HighLevelCommand.STRAIGHT else 0.0

        direction, _angle, offset = self._lane_estimate(image)
        drivable_off = None
        if direction != "NONE":
            self._last_lane_offset_norm = float(np.clip(offset, -1.0, 1.0))
        elif drivable_off is not None:
            self._last_lane_offset_norm = float(np.clip(drivable_off, -1.0, 1.0))
        lane_offset_norm = self._last_lane_offset_norm
        if direction != "NONE":
            self._last_is_on_road = 1.0
        wp = self.world.get_map().get_waypoint(self.ego.get_transform().location)
        is_on_road = (
            1.0 if (wp is not None and wp.is_junction) else self._last_is_on_road
        )

        if self._use_ground_truth_lane:
            lane_offset_norm, is_on_road = self._ground_truth_lane(wp)
            self._last_lane_offset_norm = lane_offset_norm

        objects, _ = self.perception.perceive(image)
        self.last_objects = objects

        image_width = float(image.shape[1])
        vehicles_ahead = [
            o
            for o in objects
            if o.class_name == ObjectClass.VEHICLE and _in_ego_path(o, image_width)
        ]
        red_lights_ahead = [
            o
            for o in objects
            if o.class_name == ObjectClass.RED_LIGHT and _in_ego_path(o, image_width)
        ]

        nearest_vehicle_norm = _nearest_distance_norm(
            vehicles_ahead, (ObjectClass.VEHICLE,)
        )
        red_light_distance_norm = _nearest_distance_norm(
            red_lights_ahead, (ObjectClass.RED_LIGHT,)
        )
        nearest_walker_norm = _nearest_distance_norm(objects, (ObjectClass.WALKER,))
        nearest_stop_yield_norm = _nearest_distance_norm(
            objects, (ObjectClass.STOP, ObjectClass.YIELD)
        )

        speed_signs = [o for o in objects if o.class_name in _SPEED_LIMIT_KMH]
        if speed_signs:
            self._current_speed_limit_kmh = _SPEED_LIMIT_KMH[speed_signs[0].class_name]
        speed_limit_norm = float(
            np.clip(self._current_speed_limit_kmh / _MAX_SPEED_KMH, 0.0, 1.0)
        )

        obs = [
            speed_norm,
            cmd_left,
            cmd_right,
            cmd_straight,
            lane_offset_norm,
            is_on_road,
            nearest_vehicle_norm,
            red_light_distance_norm,
            speed_limit_norm,
            nearest_walker_norm,
            nearest_stop_yield_norm,
            self._prev_steer,
            self._prev_accel,
        ]
        if self._use_goal_bearing:
            target = self._route_lookahead_target()
            obs.append(
                _signed_bearing_norm(
                    v_loc.x, v_loc.y, v_transform.rotation.yaw, target.x, target.y
                )
                if target is not None
                else 0.0
            )
        return np.array(obs, dtype=np.float32)

    def render(self) -> np.ndarray | None:
        return self._last_image

    def _teleport_to_spawn(self, spawn_idx: int | None = None) -> None:
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            return
        if spawn_idx is not None:
            idx = int(spawn_idx) % len(spawn_points)
            spawn = spawn_points[idx]
        else:
            spawn = self._pick_safe_random_spawn(spawn_points)
        self.ego.set_transform(spawn)
        try:
            from carla import Vector3D, VehicleControl  # noqa: PLC0415

            self.ego.set_target_velocity(Vector3D(0, 0, 0))
            self.ego.apply_control(VehicleControl())
        except ModuleNotFoundError:
            self.ego.set_target_velocity(None)
            self.ego.apply_control(None)

    def _pick_safe_random_spawn(self, spawn_points: list) -> "carla.Transform":
        """First random spawn clear of nearby actors, else the best seen."""
        actors = self._nearby_dynamic_actors()
        best_spawn = None
        best_dist = -1.0
        for _ in range(_SPAWN_SAFETY_MAX_ATTEMPTS):
            candidate = spawn_points[int(self.np_random.integers(len(spawn_points)))]
            dist = self._min_actor_distance(candidate.location, actors)
            if dist >= _SPAWN_SAFETY_MIN_DIST_M:
                return candidate
            if dist > best_dist:
                best_dist = dist
                best_spawn = candidate
        return best_spawn

    def _ground_truth_lane(self, wp) -> tuple[float, float]:
        """Ground-truth (lane_offset_norm, is_on_road) from CARLA's map (diagnostic)."""
        import carla

        loc = self.ego.get_transform().location
        on_wp = self.world.get_map().get_waypoint(
            loc, project_to_road=False, lane_type=carla.LaneType.Driving
        )
        is_on_road = 1.0 if on_wp is not None else 0.0
        if wp is None:
            return self._last_lane_offset_norm, is_on_road
        right = wp.transform.get_right_vector()
        offset_norm = _signed_lane_offset_norm(
            loc.x,
            loc.y,
            wp.transform.location.x,
            wp.transform.location.y,
            right.x,
            right.y,
            wp.lane_width,
        )
        return offset_norm, is_on_road

    def _maybe_advance_curriculum(self) -> None:
        """Raise the destination-distance ceiling when recent success clears the bar."""
        if (
            len(self._recent_success) >= _CURRICULUM_WINDOW
            and sum(self._recent_success) / len(self._recent_success)
            >= _CURRICULUM_ADVANCE_RATE
            and self._curriculum_max_m < _CURRICULUM_CEILING_M
        ):
            self._curriculum_max_m = min(
                self._curriculum_max_m + _CURRICULUM_STEP_M, _CURRICULUM_CEILING_M
            )
            self._recent_success.clear()
            print(f"    curriculum: destination max -> {self._curriculum_max_m:.0f} m")

    def _pick_random_destination(
        self, spawn_pts: list, ego_location: "carla.Location"
    ) -> "carla.Location":
        """Random destination within the current curriculum distance window."""
        best_dest = None
        best_gap = float("inf")
        for _ in range(_DEST_PICK_MAX_ATTEMPTS):
            candidate = spawn_pts[int(self.np_random.integers(len(spawn_pts)))]
            dist = ego_location.distance(candidate.location)
            if _CURRICULUM_MIN_FLOOR_M <= dist <= self._curriculum_max_m:
                return candidate.location
            gap = (
                _CURRICULUM_MIN_FLOOR_M - dist
                if dist < _CURRICULUM_MIN_FLOOR_M
                else dist - self._curriculum_max_m
            )
            if gap < best_gap:
                best_gap = gap
                best_dest = candidate.location
        return best_dest

    def _nearby_dynamic_actors(self) -> list:
        """All vehicle and pedestrian actors in the world, excluding the ego."""
        actors = self.world.get_actors()
        vehicles = list(actors.filter("vehicle.*"))
        walkers = list(actors.filter("walker.pedestrian.*"))
        return [a for a in vehicles + walkers if a.id != self.ego.id]

    @staticmethod
    def _min_actor_distance(location: "carla.Location", actors: list) -> float:
        """Distance from `location` to the nearest actor, or inf if none."""
        if not actors:
            return float("inf")
        return min(location.distance(a.get_location()) for a in actors)

    def _apply_control(self, steer: float, throttle: float, brake: float) -> None:
        try:
            from carla import VehicleControl  # noqa: PLC0415

            control = VehicleControl(throttle=throttle, steer=steer, brake=brake)
        except ModuleNotFoundError:
            control = None
        self.ego.apply_control(control)

    def _speed_kmh(self) -> float:
        v = self.ego.get_velocity()
        return math.sqrt(v.x**2 + v.y**2 + v.z**2) * 3.6

    def _dist_to_destination(self) -> float | None:
        """Straight-line 2D distance from ego to the route destination, or None."""
        if self.route is None or self.route.destination is None:
            return None
        loc = self.ego.get_transform().location
        dest = self.route.destination
        return math.sqrt((loc.x - dest.x) ** 2 + (loc.y - dest.y) ** 2)

    def _reached_destination(self) -> bool:
        """True once ego is near the destination and has driven a minimum distance."""
        dist_to_dest = self._dist_to_destination()
        if dist_to_dest is None or self._episode_start_location is None:
            return False
        loc = self.ego.get_transform().location
        dist_travelled = math.sqrt(
            (loc.x - self._episode_start_location.x) ** 2
            + (loc.y - self._episode_start_location.y) ** 2
        )
        return (
            dist_to_dest < _DEST_REACHED_RADIUS_M
            and dist_travelled >= _MIN_TRAVEL_FOR_DEST_M
        )

    def _update_nearest_route_idx(self) -> float:
        """Slide _route_idx to the nearest waypoint; return the squared distance."""
        if not (self.route and self.route.waypoints):
            return float("inf")
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
        return min_dist_sq

    def _is_off_route(self) -> bool:
        """True if ego is more than _OFF_ROUTE_M from the nearest route waypoint."""
        if not (self.route and self.route.waypoints):
            return False
        return self._update_nearest_route_idx() > _OFF_ROUTE_M**2

    def _route_lookahead_target(self) -> "Waypoint | None":
        """The lookahead route waypoint the bearing obs aims at."""
        if self.route is None:
            return None
        if not self.route.waypoints:
            return self.route.destination
        self._update_nearest_route_idx()
        wps = self.route.waypoints
        target_idx = min(
            self._route_idx + self._goal_bearing_lookahead_wps, len(wps) - 1
        )
        return wps[target_idx]

    def _compute_route_cumulative(self) -> None:
        """Precompute cumulative arc-length along the route waypoints (per reset)."""
        wps = self.route.waypoints if self.route else None
        if not wps or len(wps) < 2:
            self._route_cum = None
            self._route_total = None
            return
        cum = [0.0]
        for j in range(1, len(wps)):
            cum.append(
                cum[-1] + math.hypot(wps[j].x - wps[j - 1].x, wps[j].y - wps[j - 1].y)
            )
        self._route_cum = cum
        self._route_total = max(cum[-1], 1.0)

    def _route_remaining_dist(self) -> float:
        """Distance still to travel along the route to the destination."""
        wps = self.route.waypoints
        if self._route_cum is None or len(self._route_cum) != len(wps):
            self._compute_route_cumulative()
            if self._route_total is None:
                return 0.0
        i = min(self._route_idx, len(self._route_cum) - 1)
        ego = self.ego.get_transform().location
        dist_to_wp = math.hypot(ego.x - wps[i].x, ego.y - wps[i].y)
        return dist_to_wp + (self._route_total - self._route_cum[i])

    def _on_collision(self, event) -> None:
        self._collision_flag = True
        self._collision_speed_kmh = self._speed_kmh()

    def _on_camera(self, raw_image) -> None:
        arr = np.frombuffer(raw_image.raw_data, dtype=np.uint8).reshape(
            raw_image.height, raw_image.width, 4
        )
        self._last_image = arr[:, :, [2, 1, 0]]
