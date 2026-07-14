"""Smoke tests for CarlaEnv — no CARLA, no GPU."""

from __future__ import annotations

import math
from unittest.mock import Mock

import numpy as np
import pytest

from src.ai.training.rl_env import CarlaEnv, _in_ego_path
from src.interfaces.navigation_types import HighLevelCommand, Route, Waypoint
from src.interfaces.perception_types import DetectedObject, ObjectClass

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_env(
    cmd: HighLevelCommand = HighLevelCommand.LANE_FOLLOW,
    speed_mps: float = 0.0,
    lane_angle: float = 0.0,
    lane_offset: float = 0.0,
    on_road: bool = True,
    nearest_vehicle_m: float = 50.0,
    red_light_distance_m: float | None = None,
    nearest_walker_m: float | None = None,
    nearest_stop_yield_m: float | None = None,
    max_episode_steps: int = 10,
) -> CarlaEnv:
    world = Mock()
    ego = Mock()
    nav = Mock()
    camera = Mock()
    col_sensor = Mock()

    vel = Mock()
    vel.x, vel.y, vel.z = speed_mps, 0.0, 0.0
    ego.get_velocity.return_value = vel

    loc = Mock()
    loc.x, loc.y, loc.z = 0.0, 0.0, 0.0
    _add_real_distance(loc)  # _pick_random_destination() calls loc.distance(...)
    transform = Mock()
    transform.location = loc
    transform.rotation.yaw = 0.0
    ego.get_transform.return_value = transform

    wp = Mock()
    wp.transform.rotation.yaw = 0.0
    wp.is_junction = False
    world.get_map.return_value.get_waypoint.return_value = wp

    spawn = Mock()
    spawn.location.x, spawn.location.y, spawn.location.z = 0.0, 0.0, 0.0
    world.get_map.return_value.get_spawn_points.return_value = [spawn, spawn]
    world.get_actors.return_value.filter.return_value = []  # no nearby NPCs by default

    nav.next_command.return_value = cmd
    nav.plan.return_value = Route(
        waypoints=[], destination=Waypoint(0.0, 0.0, 0.0, 0.0)
    )

    # Franck's perception mock
    # Vehicle/red-light bboxes are centered on a 1280-wide frame (cx=640) so
    # they land in the middle third and survive _in_ego_path() filtering —
    # tests that specifically exercise the edge-vs-center filter build their
    # own DetectedObject lists instead of using these defaults.
    perception = Mock()
    objects = [
        DetectedObject(
            class_name=ObjectClass.VEHICLE,
            bbox=(630, 0, 650, 10),
            confidence=0.9,
            distance_m=nearest_vehicle_m,
        )
    ]
    if red_light_distance_m is not None:
        objects.append(
            DetectedObject(
                class_name=ObjectClass.RED_LIGHT,
                bbox=(630, 0, 650, 30),
                confidence=0.95,
                distance_m=red_light_distance_m,
            )
        )
    if nearest_walker_m is not None:
        objects.append(
            DetectedObject(
                class_name=ObjectClass.WALKER,
                bbox=(200, 0, 220, 60),
                confidence=0.9,
                distance_m=nearest_walker_m,
            )
        )
    if nearest_stop_yield_m is not None:
        objects.append(
            DetectedObject(
                class_name=ObjectClass.STOP,
                bbox=(300, 0, 320, 40),
                confidence=0.9,
                distance_m=nearest_stop_yield_m,
            )
        )
    perception.perceive.return_value = (
        objects,
        np.zeros((720, 1280), dtype=np.float32),
    )

    # Karim's lane estimate mock
    direction = "ALIGNE" if on_road else "NONE"
    lane_estimate_fn = Mock(return_value=(direction, lane_angle, lane_offset))

    route = Route(waypoints=[], destination=Waypoint(0.0, 0.0, 0.0, 0.0))

    return CarlaEnv(
        world=world,
        ego_vehicle=ego,
        nav=nav,
        route=route,
        perception=perception,
        lane_estimate_fn=lane_estimate_fn,
        camera=camera,
        collision_sensor=col_sensor,
        max_episode_steps=max_episode_steps,
    )


def _set_ego_location(env, x: float, y: float, z: float = 0.0) -> None:
    """Overrides the mocked ego's current position for a single check."""
    loc = Mock()
    loc.x, loc.y, loc.z = x, y, z
    transform = Mock()
    transform.location = loc
    transform.rotation.yaw = 0.0
    env.ego.get_transform.return_value = transform


def _stub_ego_position_changes_after_tick(
    env, pre_xyz: tuple[float, float, float], post_xyz: tuple[float, float, float]
) -> tuple[Mock, Mock]:
    """Makes env.ego.get_transform() return a location built from `pre_xyz`
    until env.world.tick() has been called at least once, then a location
    built from `post_xyz` afterwards — simulating CARLA's real behaviour
    where set_transform() only takes effect client-side on the next
    world.tick(). Returns (pre_loc, post_loc) so callers can assert identity.
    """
    pre_loc = _add_real_distance(Mock(x=pre_xyz[0], y=pre_xyz[1], z=pre_xyz[2]))
    pre_transform = Mock(location=pre_loc)
    pre_transform.rotation.yaw = 0.0
    post_loc = _add_real_distance(Mock(x=post_xyz[0], y=post_xyz[1], z=post_xyz[2]))
    post_transform = Mock(location=post_loc)
    post_transform.rotation.yaw = 0.0

    def get_transform_side_effect():
        return post_transform if env.world.tick.called else pre_transform

    env.ego.get_transform = Mock(side_effect=get_transform_side_effect)
    return pre_loc, post_loc


def _make_spawn(distance_to_nearest: float) -> Mock:
    """A spawn-point Mock whose location reports a fixed nearest-actor distance,
    regardless of which actor is queried. Its (x, y, z) is set to a fixed point
    100m from the default (0, 0, 0) ego mock position (see _make_env()) — well
    past _MIN_DEST_DIST_M (30m) — so _pick_random_destination()'s real
    ego_location.distance(candidate.location) call (see _add_real_distance())
    succeeds with a genuine value instead of raising a TypeError that reset()'s
    broad except would otherwise silently swallow."""
    sp = Mock()
    sp.location.x, sp.location.y, sp.location.z = 100.0, 0.0, 0.0
    sp.location.distance = Mock(return_value=distance_to_nearest)
    return sp


def _add_real_distance(loc: Mock) -> Mock:
    """Attaches a `.distance()` method to a location Mock that computes real
    Euclidean distance to another x/y/z-bearing object, mirroring
    carla.Location.distance() — needed for _pick_random_destination(), which
    calls `.distance()` on the ego's own (mocked) location.
    """
    loc.distance = Mock(
        side_effect=lambda other: math.sqrt(
            (loc.x - other.x) ** 2 + (loc.y - other.y) ** 2 + (loc.z - other.z) ** 2
        )
    )
    return loc


def _make_spawn_at(x: float, y: float, z: float = 0.0) -> Mock:
    """A spawn-point Mock at a specific, distinguishable (x, y, z) — for
    destination-selection tests that need distinct candidate locations."""
    sp = Mock()
    sp.location.x, sp.location.y, sp.location.z = x, y, z
    return sp


_ZERO_ACTION = np.array([0.0, 0.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Spaces
# ---------------------------------------------------------------------------


def test_observation_space_shape_and_dtype():
    env = _make_env()
    assert env.observation_space.shape == (13,)
    assert env.observation_space.dtype == np.float32


def test_observation_space_grows_to_13_scalars():
    env = _make_env()
    assert env.observation_space.shape == (13,)
    np.testing.assert_array_equal(env.observation_space.low[11:13], [-1.0, -1.0])
    np.testing.assert_array_equal(env.observation_space.high[11:13], [1.0, 1.0])


def test_obs_prev_steer_and_prev_accel_start_at_zero_after_reset():
    env = _make_env()
    obs, _ = env.reset()
    assert obs[11] == pytest.approx(0.0)
    assert obs[12] == pytest.approx(0.0)


def test_obs_prev_steer_and_prev_accel_reflect_last_action():
    env = _make_env()
    env.reset()
    obs, _, _, _, _ = env.step(np.array([0.4, -0.6], dtype=np.float32))
    assert obs[11] == pytest.approx(0.4)
    assert obs[12] == pytest.approx(-0.6)


def test_action_space_shape_and_bounds():
    env = _make_env()
    assert env.action_space.shape == (2,)
    np.testing.assert_array_equal(env.action_space.low, [-1.0, -1.0])
    np.testing.assert_array_equal(env.action_space.high, [1.0, 1.0])


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


def test_reset_returns_obs_and_empty_info():
    obs, info = _make_env().reset()
    assert isinstance(obs, np.ndarray)
    assert info == {}


def test_reset_obs_shape_and_dtype():
    obs, _ = _make_env().reset()
    assert obs.shape == (13,)
    assert obs.dtype == np.float32


def test_reset_obs_within_observation_space():
    env = _make_env()
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)


def test_reset_clears_collision_flag():
    env = _make_env()
    env._collision_flag = True
    env.reset()
    assert env._collision_flag is False


def test_reset_teleports_ego():
    env = _make_env()
    env.reset()
    env.ego.set_transform.assert_called()
    env.ego.set_target_velocity.assert_called()


def test_teleport_resets_vehicle_control():
    # Regression test: a previous episode ending at full throttle (e.g.
    # mid-collision) must not leave that control active on the actuator
    # through the new episode's warmup ticks. Only asserts the call
    # happened (not the exact argument), mirroring how
    # set_target_velocity's own call is asserted just above — the real
    # `carla` package is importable in this environment, so this exercises
    # the real-CARLA branch (an actual zeroed VehicleControl), not the
    # ModuleNotFoundError fallback.
    env = _make_env()
    env.ego.apply_control.reset_mock()
    env.reset()
    env.ego.apply_control.assert_called()


def test_reset_replans_route_even_without_explicit_spawn_idx():
    # Regression test: SB3 calls reset() with no options after every training
    # episode (spawn_idx=None). The route must still be replanned from the
    # new (randomly teleported) position — otherwise nav commands are computed
    # against a stale route from a completely different location.
    env = _make_env()
    env.nav.plan.reset_mock()
    env.reset()
    env.nav.plan.assert_called_once()


def test_reset_replans_route_with_explicit_spawn_idx():
    env = _make_env()
    env.nav.plan.reset_mock()
    env.reset(options={"spawn_idx": 1})
    env.nav.plan.assert_called_once()


# ---------------------------------------------------------------------------
# Safe spawn selection
# ---------------------------------------------------------------------------


def test_safe_spawn_rejects_candidate_within_min_distance():
    env = _make_env()
    unsafe = _make_spawn(distance_to_nearest=5.0)
    safe = _make_spawn(distance_to_nearest=15.0)
    env.world.get_map.return_value.get_spawn_points.return_value = [unsafe, safe]
    nearby_actor = Mock(id=999, get_location=Mock(return_value=Mock()))
    env.world.get_actors.return_value.filter.return_value = [nearby_actor]
    # 3 draws total: 2 for the safe-spawn pick (unsafe, then safe) and 1 more
    # for the random destination pick that follows in the same reset() call,
    # which succeeds immediately since _make_spawn()'s candidates sit 100m
    # from the ego's mocked (0, 0, 0) position.
    env.np_random = Mock(integers=Mock(side_effect=[0, 1, 0]))

    env.reset()

    env.ego.set_transform.assert_called_with(safe)


def test_safe_spawn_accepts_first_candidate_at_exactly_min_distance():
    env = _make_env()
    safe = _make_spawn(distance_to_nearest=10.0)  # exactly the threshold
    other = _make_spawn(distance_to_nearest=20.0)
    env.world.get_map.return_value.get_spawn_points.return_value = [safe, other]
    nearby_actor = Mock(id=999, get_location=Mock(return_value=Mock()))
    env.world.get_actors.return_value.filter.return_value = [nearby_actor]
    env.np_random = Mock(integers=Mock(side_effect=[0, 0]))

    env.reset()

    env.ego.set_transform.assert_called_with(safe)
    # 2 draws total: one for the safe-spawn pick (safe's 10.0m clearance
    # meets the 10m threshold on the first draw) and one for the random
    # destination pick that follows (reset() shares self.np_random across
    # both). The destination draw also succeeds on its first attempt:
    # _make_spawn()'s candidates sit 100m from the ego's mocked (0, 0, 0)
    # position, comfortably past the 30m _MIN_DEST_DIST_M guard, so
    # ego_location.distance(...) returns a real value >= 30 immediately
    # rather than needing a retry.
    assert env.np_random.integers.call_count == 2


def test_safe_spawn_falls_back_to_best_attempt_when_all_unsafe():
    env = _make_env()
    worse = _make_spawn(distance_to_nearest=2.0)
    better = _make_spawn(distance_to_nearest=6.0)
    env.world.get_map.return_value.get_spawn_points.return_value = [worse, better]
    nearby_actor = Mock(id=999, get_location=Mock(return_value=Mock()))
    env.world.get_actors.return_value.filter.return_value = [nearby_actor]
    # 10 draws (all below the 10m threshold), ending on `worse` so a
    # last-drawn-wins implementation would incorrectly return `worse`, plus
    # 1 more draw for the random destination pick that follows.
    env.np_random = Mock(integers=Mock(side_effect=[0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]))

    env.reset()

    env.ego.set_transform.assert_called_with(better)


def test_safe_spawn_accepts_immediately_when_no_nearby_actors():
    env = _make_env()
    only = _make_spawn(distance_to_nearest=0.0)  # would be unsafe if actors existed
    env.world.get_map.return_value.get_spawn_points.return_value = [only]
    env.world.get_actors.return_value.filter.return_value = (
        []
    )  # no vehicles, no walkers
    # 2 draws: 1 for the safe-spawn pick (immediate, no nearby actors) and 1
    # for the random destination pick that follows.
    env.np_random = Mock(integers=Mock(side_effect=[0, 0]))

    env.reset()

    env.ego.set_transform.assert_called_with(only)


def test_safe_spawn_excludes_ego_from_nearby_actors():
    env = _make_env()
    safe = _make_spawn(distance_to_nearest=999.0)
    env.world.get_map.return_value.get_spawn_points.return_value = [safe]
    env.ego.id = 1
    ego_as_actor = Mock(id=1, get_location=Mock(return_value=Mock()))
    env.world.get_actors.return_value.filter.return_value = [ego_as_actor]
    # 2 draws: 1 for the safe-spawn pick (immediate, ego excluded from
    # nearby actors) and 1 for the random destination pick that follows.
    env.np_random = Mock(integers=Mock(side_effect=[0, 0]))

    env.reset()

    env.ego.set_transform.assert_called_with(safe)
    safe.location.distance.assert_not_called()  # the only "actor" found was the ego itself


def test_explicit_spawn_idx_skips_safety_check():
    env = _make_env()
    spawn0 = _make_spawn(distance_to_nearest=0.0)
    env.world.get_map.return_value.get_spawn_points.return_value = [spawn0]

    env.reset(options={"spawn_idx": 0})

    env.world.get_actors.assert_not_called()


def test_reset_does_warmup_ticks():
    env = _make_env()
    env.reset()
    assert env.world.tick.call_count >= 5


def test_reset_clears_last_image():
    env = _make_env()
    env._last_image = np.zeros((720, 1280, 3), dtype=np.uint8)
    env.reset()
    assert env._last_image is None


def test_reset_plans_route_from_post_tick_ego_position():
    # Regression test: set_transform() only takes effect client-side on the
    # next world.tick(). nav.plan() must be called with the ego's position
    # as read AFTER the warmup ticks, not the stale pre-teleport position.
    env = _make_env()
    env.nav.plan.reset_mock()
    _pre_loc, post_loc = _stub_ego_position_changes_after_tick(
        env, pre_xyz=(1.0, 2.0, 0.0), post_xyz=(9.0, 8.0, 0.0)
    )

    env.reset()

    env.nav.plan.assert_called_once()
    called_location = env.nav.plan.call_args[0][0]
    assert called_location is post_loc


def test_reset_episode_start_location_uses_post_tick_ego_position():
    # Same root cause as above: _episode_start_location must be captured
    # after the warmup ticks, or the "reached destination" bonus can fire
    # after almost no travel (it did, in production).
    env = _make_env()
    _pre_loc, post_loc = _stub_ego_position_changes_after_tick(
        env, pre_xyz=(1.0, 2.0, 0.0), post_xyz=(9.0, 8.0, 0.0)
    )

    env.reset()

    assert env._episode_start_location is post_loc


def test_reset_clears_collision_flag_set_during_warmup_ticks():
    # Regression test: a transient collision during the settling ticks
    # (e.g. a nearby NPC clips the just-teleported vehicle) must not poison
    # the new episode. This requires _collision_flag to be reset to False
    # AFTER the warmup-tick loop, not before it.
    env = _make_env()
    env.world.tick = Mock(side_effect=lambda: setattr(env, "_collision_flag", True))

    env.reset()

    assert env._collision_flag is False


def test_reset_nav_plan_exception_does_not_propagate():
    env = _make_env()
    env.nav.plan.side_effect = RuntimeError("boom")
    env.reset()  # must not raise


def test_reset_nav_plan_exception_is_printed(capsys):
    env = _make_env()
    env.nav.plan.side_effect = RuntimeError("boom")
    env.reset()
    captured = capsys.readouterr()
    assert "boom" in captured.out


# ---------------------------------------------------------------------------
# Random destination selection
# ---------------------------------------------------------------------------


def test_reset_picks_different_destination_across_differently_seeded_episodes():
    # Regression test: reset() used to always plan toward spawn_pts[-1], the
    # same fixed destination every episode. It must now vary with the random
    # draw so the policy practices reaching different points on the map.
    env = _make_env()
    near = _make_spawn_at(5.0, 0.0)  # closer than _MIN_DEST_DIST_M
    far_a = _make_spawn_at(100.0, 0.0)
    far_b = _make_spawn_at(0.0, 100.0)
    env.world.get_map.return_value.get_spawn_points.return_value = [
        near,
        far_a,
        far_b,
    ]

    env.np_random = Mock(integers=Mock(return_value=1))  # always draws far_a
    env.reset()
    dest_a = env.nav.plan.call_args[0][1]

    env.np_random = Mock(integers=Mock(return_value=2))  # always draws far_b
    env.reset()
    dest_b = env.nav.plan.call_args[0][1]

    assert dest_a is far_a.location
    assert dest_b is far_b.location
    assert dest_a is not dest_b


def test_pick_random_destination_enforces_min_distance_guard():
    env = _make_env()
    ego_loc = _add_real_distance(Mock(x=0.0, y=0.0, z=0.0))
    near = _make_spawn_at(5.0, 0.0)  # 5m away, below the 30m guard
    far = _make_spawn_at(50.0, 0.0)  # 50m away, clears the guard
    env.np_random = Mock(integers=Mock(side_effect=[0, 1]))  # draws near, then far

    dest = env._pick_random_destination([near, far], ego_loc)

    assert dest is far.location
    assert ego_loc.distance(dest) >= 30.0  # >= _MIN_DEST_DIST_M


def test_pick_random_destination_falls_back_to_farthest_when_all_too_close():
    env = _make_env()
    ego_loc = _add_real_distance(Mock(x=0.0, y=0.0, z=0.0))
    closer = _make_spawn_at(5.0, 0.0)  # 5m, both below the 30m guard
    farther = _make_spawn_at(20.0, 0.0)  # 20m, farthest seen
    # 10 draws alternating, ending on `closer` so a last-drawn-wins
    # implementation would incorrectly return `closer`.
    env.np_random = Mock(integers=Mock(side_effect=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]))

    dest = env._pick_random_destination([closer, farther], ego_loc)

    assert dest is farther.location
    assert dest is not None


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------


def test_step_returns_five_tuple_correct_types():
    env = _make_env()
    env.reset()
    obs, reward, terminated, truncated, info = env.step(_ZERO_ACTION)
    assert obs.shape == (13,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_collision_terminates_with_penalty():
    env = _make_env()
    env.reset()
    env._collision_flag = True
    _, reward, terminated, _, _ = env.step(_ZERO_ACTION)
    assert terminated is True
    assert reward == pytest.approx(-5.0)


def test_collision_captures_impact_speed():
    env = _make_env(speed_mps=20.0)  # 72 km/h
    env.reset()
    env._on_collision(Mock())
    assert env._collision_speed_kmh == pytest.approx(72.0, abs=0.1)


def test_collision_speed_resets_on_reset():
    env = _make_env(speed_mps=20.0)
    env.reset()
    env._on_collision(Mock())
    env.reset()
    assert env._collision_speed_kmh == pytest.approx(0.0)


def test_red_light_violation_penalized_once_not_twice():
    env = _make_env(
        speed_mps=10.0, red_light_distance_m=3.0
    )  # 36 km/h, close red light
    env.reset()
    _, reward1, _, _, _ = env.step(_ZERO_ACTION)
    _, reward2, _, _, _ = env.step(_ZERO_ACTION)
    # Expected diff is 2.0 (the red-light violation penalty). This fixture's
    # mocked destination sits exactly at the ego's fixed spawn location
    # (dist0=0) and the ego never moves, so r_progress is 0.0 on every step
    # (no distance change to reward) and doesn't affect the diff.
    assert reward2 - reward1 == pytest.approx(2.0, abs=1e-3)


def test_red_light_violation_rearms_after_moving_away():
    env = _make_env(speed_mps=10.0, red_light_distance_m=3.0)
    env.reset()
    env.step(_ZERO_ACTION)
    assert env._red_light_flagged is True
    far_objects = [
        DetectedObject(
            class_name=ObjectClass.RED_LIGHT,
            bbox=(0, 0, 1, 1),
            confidence=0.9,
            distance_m=40.0,
        )
    ]
    env.perception.perceive.return_value = (
        far_objects,
        np.zeros((720, 1280), dtype=np.float32),
    )
    env.step(_ZERO_ACTION)
    assert env._red_light_flagged is False


def test_stop_yield_violation_penalized_once_not_twice():
    env = _make_env(speed_mps=10.0, nearest_stop_yield_m=3.0)
    env.reset()
    _, reward1, _, _, _ = env.step(_ZERO_ACTION)
    _, reward2, _, _, _ = env.step(_ZERO_ACTION)
    # Expected diff is 1.0 (the stop/yield violation penalty). Same
    # zero-progress reasoning as test_red_light_violation_penalized_once_not_twice.
    assert reward2 - reward1 == pytest.approx(1.0, abs=1e-3)


def test_max_steps_truncates_episode():
    env = _make_env(max_episode_steps=3)
    env.reset()
    for _ in range(2):
        _, _, _, truncated, _ = env.step(_ZERO_ACTION)
        assert not truncated
    _, _, _, truncated, _ = env.step(_ZERO_ACTION)
    assert truncated is True


def test_reward_center_term_uses_lane_offset_when_centered():
    # Large heading angle but laterally centred → r_center should be at its max (0.3).
    env = _make_env(lane_angle=80.0, lane_offset=0.0)
    env.reset()
    _, reward, _, _, _ = env.step(_ZERO_ACTION)
    # speed=0 -> r_center=0.3, r_alive=0.05, r_stall=-0.20 (speed < 1 km/h),
    # r_safe=0.05 (no vehicle/walker/speed-limit configured, so nothing dangerous is
    # active). r_progress=0.0: this fixture's mocked destination sits exactly at
    # the ego's fixed spawn location (dist0=0) and the ego never moves, so there's
    # no distance change to reward.
    assert reward == pytest.approx(0.20, abs=1e-4)


def test_reward_center_term_uses_lane_offset_when_off_center():
    # Small heading angle but laterally off-centre → r_center should be penalised.
    env = _make_env(lane_angle=0.0, lane_offset=0.9)
    env.reset()
    _, reward, _, _, _ = env.step(_ZERO_ACTION)
    # speed=0 -> r_center=(1-0.9)*0.3=0.03, r_alive=0.05, r_stall=-0.20,
    # r_safe=0.05 (no vehicle/walker/speed-limit configured, so nothing dangerous is
    # active). r_progress=0.0, same zero-progress reasoning as
    # test_reward_center_term_uses_lane_offset_when_centered.
    assert reward == pytest.approx(-0.07, abs=1e-4)


# ---------------------------------------------------------------------------
# Observation encoding
# ---------------------------------------------------------------------------


def test_obs_nav_left_one_hot():
    env = _make_env(cmd=HighLevelCommand.LEFT)
    obs, _ = env.reset()
    # obs = [speed, cmd_left, cmd_right, cmd_straight, ...]
    assert obs[1] == pytest.approx(1.0)
    assert obs[2] == pytest.approx(0.0)
    assert obs[3] == pytest.approx(0.0)


def test_obs_nav_right_one_hot():
    env = _make_env(cmd=HighLevelCommand.RIGHT)
    obs, _ = env.reset()
    assert obs[1] == pytest.approx(0.0)
    assert obs[2] == pytest.approx(1.0)
    assert obs[3] == pytest.approx(0.0)


def test_obs_nav_straight_one_hot():
    env = _make_env(cmd=HighLevelCommand.STRAIGHT)
    obs, _ = env.reset()
    assert obs[1] == pytest.approx(0.0)
    assert obs[2] == pytest.approx(0.0)
    assert obs[3] == pytest.approx(1.0)


def test_obs_speed_normalized():
    # 10 m/s = 36.0 km/h → 36 / 90 ≈ 0.4
    env = _make_env(speed_mps=10.0)
    obs, _ = env.reset()
    assert obs[0] == pytest.approx(36.0 / 90.0, abs=1e-4)


def test_obs_lane_offset_normalized():
    env = _make_env(lane_offset=0.4)
    obs, _ = env.reset()
    assert obs[4] == pytest.approx(0.4, abs=1e-4)


def test_obs_lane_offset_clipped_to_bounds():
    env = _make_env(lane_offset=1.5)
    obs, _ = env.reset()
    assert obs[4] == pytest.approx(1.0)


def test_obs_lane_offset_holds_last_valid_reading_when_lane_lost():
    """If lane detection is lost (direction="NONE"), the last known valid
    offset must be reused instead of resetting to lane_geometry()'s NO_LANE
    default of 0.0 -- that 0.0 means "no measurement", not "centered", and
    feeding it straight to the policy falsely signals "you are centered"
    the instant the car leaves the road (runs/2026-07-12_00-45_ppo_v11_150k/
    ANALYSIS.md, Constat #2)."""
    env = _make_env(lane_offset=0.7, on_road=True)
    env.reset()
    env._lane_estimate.return_value = ("NONE", 0.0, 0.0)
    obs, *_ = env.step(_ZERO_ACTION)
    assert obs[4] == pytest.approx(0.7, abs=1e-4)


def test_obs_lane_offset_defaults_to_zero_before_any_detection():
    """Before any successful lane detection has ever happened, the held
    offset must default to 0.0 (assume centered at spawn) -- even if
    lane_geometry() itself would have reported a nonzero offset alongside
    direction="NONE" (which never happens in production, but must not leak
    through the hold-last-value logic if it did)."""
    env = _make_env(on_road=False, lane_offset=0.9)
    obs, _ = env.reset()
    assert obs[4] == pytest.approx(0.0)


def test_obs_is_on_road_when_aligned():
    env = _make_env(on_road=True)
    obs, _ = env.reset()
    assert obs[5] == pytest.approx(1.0)


def test_obs_is_off_road_when_none():
    env = _make_env(on_road=False)
    obs, _ = env.reset()
    assert obs[5] == pytest.approx(0.0)


def test_obs_is_on_road_when_in_junction_without_lane_lines():
    # Intersections have no lane markings, so Karim's lane detector reports
    # direction="NONE" there. Being inside a CARLA junction must still count
    # as on-road so legitimately crossing an intersection isn't punished.
    env = _make_env(on_road=False)
    env.world.get_map.return_value.get_waypoint.return_value.is_junction = True
    obs, _ = env.reset()
    assert obs[5] == pytest.approx(1.0)


def test_obs_nearest_vehicle_normalized():
    # 25m → 25/50 = 0.5
    env = _make_env(nearest_vehicle_m=25.0)
    obs, _ = env.reset()
    assert obs[6] == pytest.approx(0.5, abs=1e-4)


def test_obs_red_light_detected():
    env = _make_env(red_light_distance_m=10.0)
    obs, _ = env.reset()
    assert obs[7] == pytest.approx(10.0 / 50.0, abs=1e-4)


def test_obs_no_red_light():
    env = _make_env(red_light_distance_m=None)
    obs, _ = env.reset()
    assert obs[7] == pytest.approx(1.0)


def test_obs_nearest_walker_normalized():
    env = _make_env(nearest_walker_m=15.0)
    obs, _ = env.reset()
    assert obs[9] == pytest.approx(15.0 / 50.0, abs=1e-4)


def test_obs_no_walker_detected():
    env = _make_env(nearest_walker_m=None)
    obs, _ = env.reset()
    assert obs[9] == pytest.approx(1.0)


def test_obs_nearest_stop_yield_normalized():
    env = _make_env(nearest_stop_yield_m=8.0)
    obs, _ = env.reset()
    assert obs[10] == pytest.approx(8.0 / 50.0, abs=1e-4)


def test_obs_no_stop_yield_detected():
    env = _make_env(nearest_stop_yield_m=None)
    obs, _ = env.reset()
    assert obs[10] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _in_ego_path() — coarse "roughly ahead of the ego" heuristic
# ---------------------------------------------------------------------------


def test_in_ego_path_excludes_bbox_centered_at_left_edge():
    obj = DetectedObject(
        class_name=ObjectClass.VEHICLE,
        bbox=(0, 100, 50, 200),
        confidence=0.9,
        distance_m=5.0,
    )
    assert _in_ego_path(obj, image_width=1280.0) is False


def test_in_ego_path_excludes_bbox_centered_at_right_edge():
    obj = DetectedObject(
        class_name=ObjectClass.VEHICLE,
        bbox=(1230, 100, 1280, 200),
        confidence=0.9,
        distance_m=5.0,
    )
    assert _in_ego_path(obj, image_width=1280.0) is False


def test_in_ego_path_includes_bbox_centered_in_middle_third():
    obj = DetectedObject(
        class_name=ObjectClass.VEHICLE,
        bbox=(600, 100, 700, 200),
        confidence=0.9,
        distance_m=5.0,
    )
    assert _in_ego_path(obj, image_width=1280.0) is True


def test_in_ego_path_includes_bbox_exactly_at_lower_third_boundary():
    # image_width=300 -> boundaries at exactly 100.0 and 200.0
    obj = DetectedObject(
        class_name=ObjectClass.VEHICLE,
        bbox=(100, 0, 100, 10),  # center == 100.0, the lower boundary
        confidence=0.9,
        distance_m=5.0,
    )
    assert _in_ego_path(obj, image_width=300.0) is True


def test_in_ego_path_includes_bbox_exactly_at_upper_third_boundary():
    obj = DetectedObject(
        class_name=ObjectClass.VEHICLE,
        bbox=(200, 0, 200, 10),  # center == 200.0, the upper boundary
        confidence=0.9,
        distance_m=5.0,
    )
    assert _in_ego_path(obj, image_width=300.0) is True


def test_in_ego_path_excludes_bbox_just_below_lower_third_boundary():
    obj = DetectedObject(
        class_name=ObjectClass.VEHICLE,
        bbox=(99, 0, 99, 10),  # center == 99.0, just outside the 100.0 boundary
        confidence=0.9,
        distance_m=5.0,
    )
    assert _in_ego_path(obj, image_width=300.0) is False


def test_in_ego_path_excludes_bbox_just_above_upper_third_boundary():
    obj = DetectedObject(
        class_name=ObjectClass.VEHICLE,
        bbox=(201, 0, 201, 10),  # center == 201.0, just outside the 200.0 boundary
        confidence=0.9,
        distance_m=5.0,
    )
    assert _in_ego_path(obj, image_width=300.0) is False


# ---------------------------------------------------------------------------
# Vehicle/red-light ego-path filtering in _get_obs()
# ---------------------------------------------------------------------------


def test_obs_excludes_vehicle_bbox_at_frame_edge():
    env = _make_env()
    env.perception.perceive.return_value = (
        [
            DetectedObject(
                class_name=ObjectClass.VEHICLE,
                bbox=(0, 100, 50, 200),  # far left edge on a 1280-wide frame
                confidence=0.9,
                distance_m=2.0,  # closest possible, but off to the side
            )
        ],
        np.zeros((720, 1280), dtype=np.float32),
    )
    obs, _ = env.reset()
    assert obs[6] == pytest.approx(1.0)  # excluded -> "nothing detected" default


def test_obs_includes_vehicle_bbox_in_middle_third():
    env = _make_env()
    env.perception.perceive.return_value = (
        [
            DetectedObject(
                class_name=ObjectClass.VEHICLE,
                bbox=(600, 100, 700, 200),  # centered ahead on a 1280-wide frame
                confidence=0.9,
                distance_m=20.0,
            )
        ],
        np.zeros((720, 1280), dtype=np.float32),
    )
    obs, _ = env.reset()
    assert obs[6] == pytest.approx(20.0 / 50.0, abs=1e-4)


def test_obs_prefers_farther_centered_vehicle_over_closer_edge_vehicle():
    env = _make_env()
    env.perception.perceive.return_value = (
        [
            DetectedObject(
                class_name=ObjectClass.VEHICLE,
                bbox=(0, 100, 50, 200),  # closer by distance, but off to the edge
                confidence=0.9,
                distance_m=2.0,
            ),
            DetectedObject(
                class_name=ObjectClass.VEHICLE,
                bbox=(600, 100, 700, 200),  # farther, but centered ahead
                confidence=0.9,
                distance_m=20.0,
            ),
        ],
        np.zeros((720, 1280), dtype=np.float32),
    )
    obs, _ = env.reset()
    assert obs[6] == pytest.approx(20.0 / 50.0, abs=1e-4)


def test_obs_excludes_red_light_bbox_at_frame_edge():
    env = _make_env()
    env.perception.perceive.return_value = (
        [
            DetectedObject(
                class_name=ObjectClass.RED_LIGHT,
                bbox=(1230, 0, 1280, 30),  # far right edge, e.g. a cross-street signal
                confidence=0.95,
                distance_m=3.0,
            )
        ],
        np.zeros((720, 1280), dtype=np.float32),
    )
    obs, _ = env.reset()
    assert obs[7] == pytest.approx(1.0)  # excluded -> "nothing detected" default


def test_obs_includes_red_light_bbox_in_middle_third():
    env = _make_env()
    env.perception.perceive.return_value = (
        [
            DetectedObject(
                class_name=ObjectClass.RED_LIGHT,
                bbox=(600, 0, 700, 30),
                confidence=0.95,
                distance_m=10.0,
            )
        ],
        np.zeros((720, 1280), dtype=np.float32),
    )
    obs, _ = env.reset()
    assert obs[7] == pytest.approx(10.0 / 50.0, abs=1e-4)


def test_obs_walker_at_frame_edge_still_included():
    # Confirms the ego-path filter is NOT applied to walkers: pedestrian
    # safety must not depend on being centered in the frame.
    env = _make_env()
    env.perception.perceive.return_value = (
        [
            DetectedObject(
                class_name=ObjectClass.WALKER,
                bbox=(0, 100, 50, 200),  # far left edge
                confidence=0.9,
                distance_m=12.0,
            )
        ],
        np.zeros((720, 1280), dtype=np.float32),
    )
    obs, _ = env.reset()
    assert obs[9] == pytest.approx(12.0 / 50.0, abs=1e-4)


# ---------------------------------------------------------------------------
# render()
# ---------------------------------------------------------------------------


def test_render_returns_none_before_camera_frame():
    env = _make_env()
    env.reset()
    result = env.render()
    assert result is None


def test_render_returns_rgb_array_after_image_set():
    env = _make_env()
    env.reset()
    env._last_image = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = env.render()
    assert isinstance(result, np.ndarray)
    assert result.shape == (720, 1280, 3)
    assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# Episode reward components
# ---------------------------------------------------------------------------


def test_episode_reward_components_empty_mid_episode():
    env = _make_env(speed_mps=10.0, max_episode_steps=10)
    env.reset()
    _, _, terminated, truncated, info = env.step(_ZERO_ACTION)
    assert terminated is False
    assert truncated is False
    assert info == {}


def test_episode_reward_components_reported_on_collision():
    env = _make_env(speed_mps=10.0)
    env.reset()
    env._collision_flag = True
    _, _, terminated, _, info = env.step(_ZERO_ACTION)
    assert terminated is True
    assert info["r_collision"] != 0.0
    assert info["r_progress"] == pytest.approx(
        0.0
    )  # collision step zeroes every other component


def test_episode_reward_components_has_all_fifteen_keys_when_reported():
    env = _make_env(speed_mps=10.0, max_episode_steps=1)
    env.reset()
    _, _, _, truncated, info = env.step(_ZERO_ACTION)
    assert truncated is True
    expected_keys = {
        "r_progress",
        "r_center",
        "r_alive",
        "r_offroad",
        "r_stall",
        "r_off_route",
        "r_following",
        "r_walker",
        "r_speeding",
        "r_red_light",
        "r_stop_yield",
        "r_collision",
        "r_destination",
        "r_safe",
        "r_jerk",
    }
    assert set(info.keys()) == expected_keys


def test_episode_reward_components_reset_between_episodes():
    env = _make_env(speed_mps=10.0, max_episode_steps=2)
    env.reset()
    env.step(_ZERO_ACTION)
    _, _, _, truncated, _ = env.step(_ZERO_ACTION)
    assert truncated is True
    env.reset()
    assert env._episode_reward_components["r_progress"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Destination check, jerk tracking
# ---------------------------------------------------------------------------


def test_reached_destination_false_when_close_but_not_travelled():
    env = _make_env()
    env.reset()  # captures start location at the default mocked (0, 0, 0)
    # Route is assigned after reset(): reset() unconditionally replans the
    # route from the (mocked) nav, which would otherwise clobber a route
    # assigned beforehand.
    env.route = Route(
        waypoints=[], destination=Waypoint(x=5.0, y=0.0, z=0.0, yaw_deg=0.0)
    )
    _set_ego_location(
        env, 5.0, 0.0
    )  # 5m from dest (< 15m radius), only 5m travelled (< 25m)
    assert env._reached_destination() is False


def test_reached_destination_false_when_travelled_but_far_from_dest():
    env = _make_env()
    env.reset()
    env.route = Route(
        waypoints=[], destination=Waypoint(x=500.0, y=0.0, z=0.0, yaw_deg=0.0)
    )
    _set_ego_location(
        env, 100.0, 0.0
    )  # 100m travelled (>= 25m), but 400m from dest (>= 15m)
    assert env._reached_destination() is False


def test_reached_destination_true_when_both_conditions_met():
    env = _make_env()
    env.reset()
    env.route = Route(
        waypoints=[], destination=Waypoint(x=100.0, y=0.0, z=0.0, yaw_deg=0.0)
    )
    _set_ego_location(env, 95.0, 0.0)  # 95m travelled (>= 25m), 5m from dest (< 15m)
    assert env._reached_destination() is True


def test_step_terminates_on_reached_destination():
    env = _make_env()
    env.reset()
    env.route = Route(
        waypoints=[], destination=Waypoint(x=100.0, y=0.0, z=0.0, yaw_deg=0.0)
    )
    _set_ego_location(env, 95.0, 0.0)
    _, reward, terminated, truncated, info = env.step(_ZERO_ACTION)
    assert terminated is True
    assert reward == pytest.approx(10.0)
    assert info["r_destination"] == pytest.approx(10.0)


def test_step_reports_positive_accumulated_r_progress_when_moving_closer():
    """Two-step progress-shaping integration test through the real reset()/step()
    code path (not a reimplementation of the formula): the destination is set
    far away via nav.plan() *before* reset() so _dist_to_dest_initial is
    computed against it. First step: no motion, establishes the baseline.
    Second step: the ego mock moves 500m closer to the destination -- the
    accumulated r_progress must be positive."""
    env = _make_env(max_episode_steps=2)
    env.nav.plan.return_value = Route(
        waypoints=[], destination=Waypoint(x=1000.0, y=0.0, z=0.0, yaw_deg=0.0)
    )
    env.reset()
    env.step(_ZERO_ACTION)  # no movement -> baseline progress_delta near zero
    _set_ego_location(env, 500.0, 0.0)  # 500m closer to the destination
    _, _, _, truncated, info = env.step(_ZERO_ACTION)
    assert truncated is True
    assert info["r_progress"] > 0.0


def test_collision_reward_scales_with_remaining_route_fraction():
    """A collision far from the destination (most of the route still ahead)
    must cost more than an otherwise-identical collision right next to it --
    integration check that CarlaEnv.step() actually threads remaining_frac
    through to compute_reward() using the same dist_norm already computed
    for r_progress."""
    env_early = _make_env(max_episode_steps=2)
    env_early.nav.plan.return_value = Route(
        waypoints=[], destination=Waypoint(x=1000.0, y=0.0, z=0.0, yaw_deg=0.0)
    )
    env_early.reset()  # ego at (0,0), dist_to_dest_initial ~= 1000m
    env_early._collision_flag = True
    _, reward_early, _, _, _ = env_early.step(_ZERO_ACTION)  # still ~1000m out

    env_late = _make_env(max_episode_steps=2)
    env_late.nav.plan.return_value = Route(
        waypoints=[], destination=Waypoint(x=1000.0, y=0.0, z=0.0, yaw_deg=0.0)
    )
    env_late.reset()
    _set_ego_location(env_late, 999.0, 0.0)  # 1m from the destination
    env_late._collision_flag = True
    _, reward_late, _, _, _ = env_late.step(_ZERO_ACTION)

    assert reward_early < reward_late


def test_prev_steer_tracks_last_action_and_resets():
    env = _make_env()
    env.reset()
    assert env._prev_steer == pytest.approx(0.0)
    env.step(np.array([0.3, 0.0], dtype=np.float32))
    assert env._prev_steer == pytest.approx(0.3)
    env.step(np.array([-0.2, 0.0], dtype=np.float32))
    assert env._prev_steer == pytest.approx(-0.2)
    env.reset()
    assert env._prev_steer == pytest.approx(0.0)


def test_episode_reward_components_includes_jerk_penalty():
    env = _make_env(max_episode_steps=2)
    env.reset()
    env.step(np.array([1.0, 0.0], dtype=np.float32))  # steer 0.0 -> 1.0, delta=1.0
    _, _, _, truncated, info = env.step(
        np.array([1.0, 0.0], dtype=np.float32)
    )  # 1.0 -> 1.0, delta=0.0
    assert truncated is True
    # r_jerk = -delta * 0.1, summed: step1 -0.1 + step2 0.0
    assert info["r_jerk"] == pytest.approx(-0.1, abs=1e-3)


# ---------------------------------------------------------------------------
# Throttle/brake derivation from accel (2D action space)
# ---------------------------------------------------------------------------


def test_step_derives_throttle_from_positive_accel():
    env = _make_env()
    env.reset()
    env.ego.apply_control.reset_mock()
    env.step(np.array([0.0, 0.7], dtype=np.float32))
    control = env.ego.apply_control.call_args[0][0]
    assert control.throttle == pytest.approx(0.7)
    assert control.brake == pytest.approx(0.0)


def test_step_derives_brake_from_negative_accel():
    env = _make_env()
    env.reset()
    env.ego.apply_control.reset_mock()
    env.step(np.array([0.0, -0.7], dtype=np.float32))
    control = env.ego.apply_control.call_args[0][0]
    assert control.throttle == pytest.approx(0.0)
    assert control.brake == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# last_objects
# ---------------------------------------------------------------------------


def test_last_objects_initialized_empty():
    env = _make_env()
    assert env.last_objects == []


def test_last_objects_populated_after_get_obs():
    env = _make_env(nearest_vehicle_m=25.0)
    env.reset()
    assert len(env.last_objects) == 1
    assert env.last_objects[0].class_name == ObjectClass.VEHICLE
    assert env.last_objects[0].distance_m == 25.0
