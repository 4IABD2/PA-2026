"""CarlaEnv — gymnasium.Env wrapper around CARLA for Phase 1 RL training."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.ai.rewards.reward_fn import compute_reward, REWARD_COMPONENT_KEYS
from src.interfaces.navigation_types import HighLevelCommand, Route, Waypoint
from src.interfaces.perception_types import ObjectClass

if TYPE_CHECKING:
    import carla
    from src.interfaces.navigation_types import Navigation
    from src.perception.pipeline import PerceptionPipeline

# obs = [speed_norm, cmd_left, cmd_right, cmd_straight,
#         lane_offset_norm, is_on_road,
#         nearest_vehicle_norm, red_light_distance_norm, speed_limit_norm,
#         nearest_walker_norm, nearest_stop_yield_norm]
_OBS_LOW = np.array(
    [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
)
_OBS_HIGH = np.array(
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32
)

_MAX_SPEED_KMH = 90.0
_MAX_OBSTACLE_M = 50.0
_WARMUP_TICKS = 5  # ticks after teleport so physics settles and sensors fill
_OFF_ROUTE_M = 15.0  # metres from nearest route waypoint before off_route penalty fires
_ROUTE_GRACE_STEPS = (
    20  # steps after reset where off-route is not penalised (car joins route)
)
_DEFAULT_SPEED_LIMIT_KMH = 30.0

_SPAWN_SAFETY_MIN_DIST_M = (
    10.0  # minimum clearance from any NPC/pedestrian for a random spawn to be "safe"
)
_SPAWN_SAFETY_MAX_ATTEMPTS = (
    10  # random spawn draws tried before falling back to the best one seen
)

_MIN_DEST_DIST_M = 30.0  # minimum straight-line distance from spawn to destination
_DEST_PICK_MAX_ATTEMPTS = (
    10  # random destination draws tried before falling back to the best one seen
)

_DEST_REACHED_RADIUS_M = 15.0  # matches _OFF_ROUTE_M's "close enough" scale
_MIN_TRAVEL_FOR_DEST_M = (
    25.0  # matches benchmark.py's _MIN_DIST_M "must have actually driven" convention
)

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
        o.distance_m
        for o in objects
        if o.class_name in classes and o.distance_m is not None
    ]
    return float(np.clip(min(dists) / _MAX_OBSTACLE_M, 0.0, 1.0)) if dists else 1.0


def _in_ego_path(obj, image_width: float) -> bool:
    """Coarse "is this roughly ahead of the ego" heuristic: True if the
    detection's bounding box is horizontally centered in the middle third
    of the camera frame. No 3D/world position is available on a detection
    to do this more precisely — this only filters out things clearly to the
    far left/right (oncoming traffic in the next lane over, cross-street
    signals, parked vehicles on the shoulder), not a real lane check.
    """
    cx = (obj.bbox[0] + obj.bbox[2]) / 2.0
    return image_width / 3.0 <= cx <= 2.0 * image_width / 3.0


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

        self.observation_space = spaces.Box(
            low=_OBS_LOW, high=_OBS_HIGH, dtype=np.float32
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
        self._last_image: np.ndarray | None = None
        self._step_count: int = 0
        self._route_idx: int = (
            0  # sliding pointer into route.waypoints for efficient off-route check
        )
        self._current_speed_limit_kmh: float = _DEFAULT_SPEED_LIMIT_KMH
        self.off_route_count: int = (
            0  # steps spent off-route this episode (readable by eval_model)
        )
        self.last_objects: list = (
            []
        )  # most recent perceive() result (readable by rl_demo for bbox overlays)

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
        # Replan route from new position so nav commands are correct at every
        # spawn. Must run unconditionally: during normal training, SB3 calls
        # reset() with no options (spawn_idx=None) after every episode, so
        # skipping this when spawn_idx is None left the route — and thus every
        # nav command fed to the policy — stale from the very first episode.
        # Runs AFTER the warmup ticks: set_transform() only takes effect
        # client-side on the next world.tick(), so reading the ego's position
        # any earlier would still see the previous episode's ending location.
        if hasattr(self.nav, "plan"):
            try:
                spawn_pts = self.world.get_map().get_spawn_points()
                dest = self._pick_random_destination(
                    spawn_pts, self.ego.get_transform().location
                )
                self.route = self.nav.plan(self.ego.get_transform().location, dest)
            except Exception as exc:
                print(f"    nav.plan failed at reset: {exc}")
        self._collision_flag = False
        self._collision_speed_kmh = 0.0
        self._red_light_flagged = False
        self._stop_yield_flagged = False
        self._episode_reward_components = {k: 0.0 for k in REWARD_COMPONENT_KEYS}
        self._episode_start_location = self.ego.get_transform().location
        self._prev_steer = 0.0
        self._step_count = 0
        self._route_idx = 0
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

        steer_delta = abs(steer - self._prev_steer)
        self._prev_steer = steer

        reward, terminated, components = compute_reward(
            speed_kmh=speed_kmh,
            center_offset=float(
                obs[4]
            ),  # lane_offset_norm, from Karim's lane detection
            is_on_road=bool(obs[5] > 0.5),  # from Karim's lane detection
            collision=self._collision_flag,
            off_route=off_route,
            nearest_vehicle_m=float(obs[6]) * _MAX_OBSTACLE_M,
            nearest_walker_m=float(obs[9]) * _MAX_OBSTACLE_M,
            red_light_distance_m=float(obs[7]) * _MAX_OBSTACLE_M,
            speed_limit_kmh=self._current_speed_limit_kmh,
            collision_speed_kmh=self._collision_speed_kmh,
            red_light_violation=red_light_violation,
            stop_yield_violation=stop_yield_violation,
            progress_speed_kmh=self._progress_speed_kmh(),
            reached_destination=self._reached_destination(),
            steer_delta=steer_delta,
        )
        for key, value in components.items():
            self._episode_reward_components[key] += value

        truncated = self._step_count >= self.max_episode_steps
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

        # Speed (onboard sensor)
        speed_norm = float(np.clip(self._speed_kmh() / _MAX_SPEED_KMH, 0.0, 1.0))

        # Navigation command from Victor's GPS
        v_transform = self.ego.get_transform()
        v_loc = v_transform.location
        vehicle_wp = Waypoint(
            x=v_loc.x, y=v_loc.y, z=v_loc.z, yaw_deg=v_transform.rotation.yaw
        )
        cmd = self.nav.next_command(vehicle_wp, self.route)
        cmd_left = 1.0 if cmd == HighLevelCommand.LEFT else 0.0
        cmd_right = 1.0 if cmd == HighLevelCommand.RIGHT else 0.0
        cmd_straight = 1.0 if cmd == HighLevelCommand.STRAIGHT else 0.0

        # Karim: lateral offset, on-road status (lane heading angle is not fed to the model)
        direction, _angle, offset = self._lane_estimate(image)
        lane_offset_norm = float(np.clip(offset, -1.0, 1.0))
        wp = self.world.get_map().get_waypoint(self.ego.get_transform().location)
        is_on_road = (
            1.0 if (direction != "NONE" or (wp is not None and wp.is_junction)) else 0.0
        )

        # Franck: nearest vehicle, red light distance, speed limit sign, walker, stop/yield
        objects, _ = self.perception.perceive(image)
        self.last_objects = objects

        # Vehicle/red-light detections are filtered to roughly "ahead of the
        # ego" (see _in_ego_path) before computing distance — otherwise the
        # nearest match anywhere in frame can be oncoming traffic in the next
        # lane, a parked car on the shoulder, or a cross-street's signal.
        # Walkers and stop/yield signs are deliberately left unfiltered.
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

        return np.array(
            [
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
            ],
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
        """Draws up to _SPAWN_SAFETY_MAX_ATTEMPTS random spawn points and returns
        the first one at least _SPAWN_SAFETY_MIN_DIST_M away from every nearby
        vehicle/pedestrian. Falls back to the candidate with the largest observed
        clearance if none clears the threshold — never worse than a pure random
        pick, and expected to essentially never trigger with light NPC traffic.
        """
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

    def _pick_random_destination(
        self, spawn_pts: list, ego_location: "carla.Location"
    ) -> "carla.Location":
        """Draws up to _DEST_PICK_MAX_ATTEMPTS random spawn points as a candidate
        destination and returns the first at least _MIN_DEST_DIST_M from the
        ego's current position — avoids a degenerate near-zero-length route.
        Falls back to the farthest candidate seen if none clears the threshold.
        """
        best_dest = None
        best_dist = -1.0
        for _ in range(_DEST_PICK_MAX_ATTEMPTS):
            candidate = spawn_pts[int(self.np_random.integers(len(spawn_pts)))]
            dist = ego_location.distance(candidate.location)
            if dist >= _MIN_DEST_DIST_M:
                return candidate.location
            if dist > best_dist:
                best_dist = dist
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
        """Distance from `location` to the nearest actor, or inf if `actors` is empty."""
        if not actors:
            return float("inf")
        return min(location.distance(a.get_location()) for a in actors)

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

    def _progress_speed_kmh(self) -> float:
        """Velocity projected onto the route's current direction, in km/h.

        Unlike _speed_kmh() (velocity magnitude), this measures progress toward
        the destination — driving fast sideways or backward relative to the
        route earns no credit. Floored at 0: driving against the route
        direction never produces a negative r_speed, only zero (off-route/
        off-road penalties already cover that failure mode separately).
        Falls back to _speed_kmh() if there's no route to project onto.
        """
        if not (self.route and self.route.waypoints):
            return self._speed_kmh()
        idx = min(self._route_idx, len(self.route.waypoints) - 1)
        yaw_rad = math.radians(self.route.waypoints[idx].yaw_deg)
        direction_x, direction_y = math.cos(yaw_rad), math.sin(yaw_rad)
        v = self.ego.get_velocity()
        projection_mps = v.x * direction_x + v.y * direction_y
        return max(0.0, projection_mps * 3.6)

    def _reached_destination(self) -> bool:
        """True once the ego is close to the route's destination AND has
        actually travelled a minimum distance this episode — the second
        condition stops a lucky spawn near the (fixed, far-away) training
        destination from collecting the bonus without driving anywhere.
        """
        # Uses `is None` rather than truthiness: Route.__len__ delegates to
        # len(waypoints), so a route with a valid destination but no waypoints
        # (the common case right after a fresh plan()) would otherwise be
        # incorrectly treated as "no route".
        if (
            self.route is None
            or self.route.destination is None
            or self._episode_start_location is None
        ):
            return False
        loc = self.ego.get_transform().location
        dest = self.route.destination
        dist_to_dest = math.sqrt((loc.x - dest.x) ** 2 + (loc.y - dest.y) ** 2)
        dist_travelled = math.sqrt(
            (loc.x - self._episode_start_location.x) ** 2
            + (loc.y - self._episode_start_location.y) ** 2
        )
        return (
            dist_to_dest < _DEST_REACHED_RADIUS_M
            and dist_travelled >= _MIN_TRAVEL_FOR_DEST_M
        )

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
        return min_dist_sq > _OFF_ROUTE_M**2

    def _on_collision(self, event) -> None:
        self._collision_flag = True
        self._collision_speed_kmh = self._speed_kmh()

    def _on_camera(self, raw_image) -> None:
        arr = np.frombuffer(raw_image.raw_data, dtype=np.uint8).reshape(
            raw_image.height, raw_image.width, 4
        )
        self._last_image = arr[:, :, [2, 1, 0]]  # BGRA → RGB
