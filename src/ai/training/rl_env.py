"""CarlaEnv — gymnasium.Env wrapper around CARLA for Phase 1 RL training."""

from __future__ import annotations

import math
from collections import deque
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
#         nearest_walker_norm, nearest_stop_yield_norm,
#         prev_steer_norm, prev_accel_norm,
#         goal_bearing_norm]   # v16: signed heading error to destination, [-1,1]
_OBS_LOW = np.array(
    [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0],
    dtype=np.float32,
)
_OBS_HIGH = np.array(
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    dtype=np.float32,
)

_MAX_SPEED_KMH = 90.0
_MAX_OBSTACLE_M = 50.0
_WARMUP_TICKS = 5  # ticks after teleport so physics settles and sensors fill
_OFF_ROUTE_M = 15.0  # metres from nearest route waypoint before off_route penalty fires
_PROGRESS_GAMMA = 0.99  # must match PPO's own gamma (src/ai/training/rl_train.py's
# _PPO_DEFAULTS) for the potential-based shaping's policy-invariance guarantee to hold
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

_DEST_PICK_MAX_ATTEMPTS = (
    10  # random destination draws tried before falling back to the best one seen
)

_DEST_REACHED_RADIUS_M = 15.0  # matches _OFF_ROUTE_M's "close enough" scale
_MIN_TRAVEL_FOR_DEST_M = (
    12.0  # must-have-actually-driven gate. Lowered 25->12 for the v14 auto-curriculum:
    # with near destinations (start window [18, 30] m) and a 15 m success radius,
    # a 25 m travel gate was unreachable (a car entering the radius of an 18-30 m
    # destination has driven far less than 25 m), so no near destination could
    # ever count as reached. 12 m still forbids collecting the +10 bonus while
    # merely drifting a few metres. Intentionally looser than benchmark.py's
    # _MIN_DIST_M for now -- training-time signal, not the benchmark criterion.
)

# --- Auto-curriculum on destination distance (v14) ------------------------
# ppo_v9..v13 never reached a single destination in *training* (0 across every
# run), so PPO never once saw the +10 terminal bonus -- it learned only to avoid
# the worst outcomes, never that arriving is possible. The fix: start
# destinations near (so the bonus is reachable and can bootstrap the value
# function) and grow the distance only as the policy earns success. Not a
# derived schedule; revisit the constants once a run actually reaches
# destinations.
_CURRICULUM_MIN_FLOOR_M = 18.0  # destinations never closer than this (with the
# 15 m success radius + 12 m travel gate, the nearest still needs ~12 m of
# directed driving -- a real "drive to it", not a drift-in-place win)
_CURRICULUM_START_MAX_M = 30.0  # initial upper bound (matches the old fixed floor)
_CURRICULUM_STEP_M = 8.0  # upper bound grows by this much per advance
_CURRICULUM_CEILING_M = 70.0  # upper bound stops growing here
_CURRICULUM_WINDOW = 20  # recent episodes considered for an advance decision
_CURRICULUM_ADVANCE_RATE = 0.5  # success rate over the window that triggers a step up

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


def _signed_lane_offset_norm(
    ego_x: float,
    ego_y: float,
    wp_x: float,
    wp_y: float,
    right_x: float,
    right_y: float,
    lane_width: float,
) -> float:
    """Signed lateral offset of the ego from a lane-centre waypoint, normalised
    to [-1, 1] by the lane half-width. Sign follows the lane's right vector
    (positive = ego is to the right of centre). Pure 2D geometry — no CARLA
    types — so it is unit-testable without a running simulator.
    """
    half_width = max(lane_width / 2.0, 0.1)
    lateral = (ego_x - wp_x) * right_x + (ego_y - wp_y) * right_y
    return float(np.clip(lateral / half_width, -1.0, 1.0))


def _signed_bearing_norm(
    ego_x: float, ego_y: float, yaw_deg: float, dest_x: float, dest_y: float
) -> float:
    """Signed heading error from the ego to a destination, normalised to
    [-1, 1] (angle / pi): 0 = destination dead ahead, ±1 = directly behind.
    Sign follows atan2 in CARLA's frame — it is consistent, so the policy
    learns the left/right mapping (the exact handedness convention does not
    matter). Pure 2D geometry — unit-testable without a CARLA runtime.
    """
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
        lane_estimate_fn: Callable[[np.ndarray], tuple[str, float, float]],
        camera: "carla.Sensor",
        collision_sensor: "carla.Sensor",
        max_episode_steps: int = 1000,
        use_ground_truth_lane: bool = False,
    ) -> None:
        super().__init__()
        self.world = world
        self.ego = ego_vehicle
        self.nav = nav
        # v15 diagnostic: when True, obs[4]/obs[5] (lane offset, on-road) come
        # from CARLA's map geometry instead of Karim's detector -- isolates
        # perception quality as the bottleneck. Default False = production path.
        self._use_ground_truth_lane = use_ground_truth_lane
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
        self._prev_accel: float = 0.0
        self._dist_to_dest_initial: float = 1.0
        self._prev_dist_to_dest_norm: float = 1.0
        self._last_lane_offset_norm: float = 0.0
        self._last_image: np.ndarray | None = None
        self._step_count: int = 0
        self._stall_steps: int = (
            0  # consecutive unjustified stall steps (ramps r_stall)
        )
        # Auto-curriculum state -- persists ACROSS episodes (never reset per
        # episode): the current destination-distance ceiling and a rolling
        # window of recent episode outcomes (True = destination reached).
        self._curriculum_max_m: float = _CURRICULUM_START_MAX_M
        self._recent_success: deque = deque(maxlen=_CURRICULUM_WINDOW)
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
        self._collision_flag = False
        self._collision_speed_kmh = 0.0
        self._red_light_flagged = False
        self._stop_yield_flagged = False
        self._episode_reward_components = {k: 0.0 for k in REWARD_COMPONENT_KEYS}
        self._episode_start_location = self.ego.get_transform().location
        self._prev_steer = 0.0
        self._prev_accel = 0.0
        self._last_lane_offset_norm = 0.0
        self._step_count = 0
        self._stall_steps = 0
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
        if dist_to_dest is not None:
            dist_norm = dist_to_dest / self._dist_to_dest_initial
            phi_now = -dist_norm
            phi_prev = -self._prev_dist_to_dest_norm
            progress_delta = _PROGRESS_GAMMA * phi_now - phi_prev
            self._prev_dist_to_dest_norm = dist_norm
        else:
            dist_norm = 1.0
            progress_delta = 0.0

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
            reached_destination=self._reached_destination(),
            steer_delta=steer_delta,
            progress_delta=progress_delta,
            remaining_frac=dist_norm,
            stall_steps=self._stall_steps,
        )
        # r_stall < 0 is exactly "this step counted as an unjustified stall":
        # legitimate stops and any movement reset the ramp.
        self._stall_steps = self._stall_steps + 1 if components["r_stall"] < 0.0 else 0
        for key, value in components.items():
            self._episode_reward_components[key] += value

        truncated = self._step_count >= self.max_episode_steps
        if terminated or truncated:
            # Record the outcome for the auto-curriculum: success == the
            # destination bonus fired this step (collision and timeout are both
            # non-successes). r_destination > 0 is the single source of truth,
            # matching how the reward itself defines "reached".
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
        if direction != "NONE":
            self._last_lane_offset_norm = float(np.clip(offset, -1.0, 1.0))
        # else: lane_geometry() couldn't track both edges and returned its
        # NO_LANE default (offset=0.0) -- that means "no measurement", not
        # "centered", so hold the last real reading instead of feeding the
        # policy a false "you're centered" signal the instant it loses the
        # road (runs/2026-07-12_00-45_ppo_v11_150k/ANALYSIS.md, Constat #2).
        lane_offset_norm = self._last_lane_offset_norm
        wp = self.world.get_map().get_waypoint(self.ego.get_transform().location)
        is_on_road = (
            1.0 if (direction != "NONE" or (wp is not None and wp.is_junction)) else 0.0
        )

        if self._use_ground_truth_lane:
            # v15 diagnostic: overwrite the two perception-derived scalars with
            # ground truth from CARLA's map. Tests whether the ~70%-off-road
            # behaviour (even in v14's *successful* episodes) is the policy being
            # fed unreliable lateral position vs a real control ceiling. Ground
            # truth is unavailable on a real car -- this localises the bottleneck,
            # it is not a shippable observation.
            lane_offset_norm, is_on_road = self._ground_truth_lane(wp)
            self._last_lane_offset_norm = lane_offset_norm

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

        # obs[13]: signed heading error to the destination (bearing-to-goal),
        # normalised to [-1, 1]. The discrete nav commands (obs[1..3]) only say
        # "turn soon", not which way the goal actually lies — this continuous
        # scalar gives the policy a direct steer-toward-goal signal (v16, to
        # attack the navigation gap left open by v15).
        dest = self.route.destination if self.route is not None else None
        if dest is not None:
            goal_bearing_norm = _signed_bearing_norm(
                v_loc.x, v_loc.y, v_transform.rotation.yaw, dest.x, dest.y
            )
        else:
            goal_bearing_norm = 0.0

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
                self._prev_steer,
                self._prev_accel,
                goal_bearing_norm,
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

    def _ground_truth_lane(self, wp) -> tuple[float, float]:
        """Ground-truth (lane_offset_norm, is_on_road) from CARLA's map, for the
        v15 diagnostic. `wp` is the nearest driving-lane waypoint (projected, as
        already fetched in _get_obs). Offset is the signed perpendicular distance
        from the ego to that lane's centre, normalised by the lane half-width;
        is_on_road is whether the ego actually sits on a driving lane (an
        unprojected query, which returns None off-lane).
        """
        # Lazy import: only runs when the experiment flag is on, so the module
        # stays importable (and unit tests runnable) without a CARLA runtime.
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
        """Step the destination-distance ceiling up once the policy clears the
        success bar over a full window of recent episodes. Called once per
        reset(), before the destination is drawn. Clearing the window on an
        advance forces success to be re-earned at the new, harder distance
        (so it can't advance every single episode off one good streak).
        """
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
        """Draws up to _DEST_PICK_MAX_ATTEMPTS random spawn points and returns the
        first whose distance falls in the current curriculum window
        [_CURRICULUM_MIN_FLOOR_M, self._curriculum_max_m] — near early on so the
        +10 destination bonus is reachable, widening as the policy earns success
        (see _maybe_advance_curriculum). Falls back to the candidate closest to
        that window if none lands inside.
        """
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

    def _dist_to_destination(self) -> float | None:
        """Straight-line 2D distance from the ego to the route's destination,
        or None if there's no route/destination to measure against.
        """
        # Uses `is None` rather than truthiness: Route.__len__ delegates to
        # len(waypoints), so a route with a valid destination but no waypoints
        # (the common case right after a fresh plan()) would otherwise be
        # incorrectly treated as "no route".
        if self.route is None or self.route.destination is None:
            return None
        loc = self.ego.get_transform().location
        dest = self.route.destination
        return math.sqrt((loc.x - dest.x) ** 2 + (loc.y - dest.y) ** 2)

    def _reached_destination(self) -> bool:
        """True once the ego is close to the route's destination AND has
        actually travelled a minimum distance this episode — the second
        condition stops a lucky spawn near the (fixed, far-away) training
        destination from collecting the bonus without driving anywhere.
        """
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
